"""Точка входа: discord.py клиент и сборка всего пайплайна.

Распределение ресурсов:

* приём (VAD + STT) параллелен по говорящим: у каждого свой буфер, свой VAD,
  сегменты уходят в пул процессов независимо;
* LLM одна на всех и обслуживается по одному запросу — узкое место CPU,
  очередь удерживает реплики от потери;
* голосовой выход один, им управляет OutputArbiter.

STT/TTS-пулы, клиент LLM и очередь общие на процесс; слушатель, арбитр и
плеер — свои на каждый сервер (guild).
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import time
from typing import Any

import discord
from discord import app_commands
from discord.ext import voice_recv

from .audio import voice_recv_patch
from .audio.listener import VoiceListener
from .config import Config, load_config
from .intent.router import Decision, Intent, IntentRouter
from .jokes import JokeTeller
from .llm.client import LlmClient
from .llm.queue_manager import ChatTask, LlmQueue
from .music import search as music_search
from .music.player import MusicPlayer
from .output.arbiter import OutputArbiter
from .stt import SttPool, Transcript, create_stt_pool
from .tts.silero_worker import TtsPool
from .utils.logging import (
    current_cid,
    guard,
    levels_path,
    publish_levels,
    setup_logging,
    stage,
    stage_summary,
    turn_drop,
    turn_note,
    turn_report,
)
from .wakeword.filter import WakeWordFilter

log = logging.getLogger(__name__)

IDLE_CHECK_INTERVAL = 30.0
#: Как часто публиковать уровни для монитора. Глазу этого хватает, а файл
#: переписывается целиком — чаще незачем.
LEVELS_INTERVAL = 0.4


class GuildSession:
    """Всё, что живёт ровно столько, сколько бот сидит в голосовом канале."""

    def __init__(self, bot: "BashmakBot", voice_client: voice_recv.VoiceRecvClient) -> None:
        self.bot = bot
        self.cfg = bot.cfg
        self.voice_client = voice_client
        self.guild = voice_client.guild
        self.channel_id = voice_client.channel.id
        self.empty_since: float | None = None

        loop = asyncio.get_running_loop()
        self.arbiter = OutputArbiter(self.cfg, loop)
        self.player = MusicPlayer(self.cfg, self.arbiter)
        self.listener = VoiceListener(self.cfg, bot.stt, self._on_transcript)

    async def start(self) -> None:
        # Один источник на всё время в канале: микшер сам разруливает,
        # что сейчас звучит — музыка, речь или их смесь.
        self.voice_client.play(self.arbiter.source)
        # listen() сам проставляет синку voice_client (AudioReader.__init__),
        # так что готовить его заранее не нужно.
        self.voice_client.listen(self.listener.sink, after=self._on_listen_stop)
        self.listener.start()
        log.info(
            "сессия запущена в #%s (%s)",
            getattr(self.voice_client.channel, "name", "?"),
            self.guild.name,
        )

    async def close(self) -> None:
        self.arbiter.interrupt()
        # Заодно снимает всё, что ещё считается для этого канала: озвучивать
        # это будет уже некому, а контекст закрытой сессии переживать не должен.
        self.bot.llm_queue.drop(self.channel_id)
        self.player.shutdown()
        await self.listener.stop()

        with contextlib.suppress(Exception):
            # У VoiceRecvClient stop() снимает и приём, и воспроизведение.
            self.voice_client.stop()
        with contextlib.suppress(Exception):
            await self.voice_client.disconnect(force=True)

        log.info("сессия в %s закрыта", self.guild.name)

    def _on_listen_stop(self, error=None, *args) -> None:  # noqa: ANN001 — сигнатура библиотеки
        """Колбэк ``after`` для listen().

        Должен быть синхронным: ``AudioReader._stop()`` зовёт его прямо из
        своего потока (``self.after(self.error)``), не через луп. Корутина
        здесь не была бы выполнена — «coroutine was never awaited».
        Первым аргументом приходит исключение, уронившее приём, или None.
        """
        if error is None:
            log.debug("приём голоса остановлен")
        else:
            log.error("приём голоса остановлен с ошибкой: %r", error)

    # ------------------------------------------------------------ речь ----
    def speaker_name(self, user_id: int) -> str:
        member = self.guild.get_member(user_id) if self.guild else None
        if member is not None:
            return member.display_name
        user = self.bot.get_user(user_id)
        return user.display_name if user is not None else f"user{user_id}"

    def silence(self) -> None:
        """Команда молчания: замолчать и выключить музыку, ничего не отвечая.

        Три источника звука гасятся по отдельности: то, что уже звучит
        (арбитр), то, что ещё считается или стоит в очереди (очередь LLM), и
        музыка. Ответ плеера («Выключил музыку») выбрасывается: команда
        замолчать не должна заканчиваться репликой.
        """
        self.arbiter.interrupt()
        self.bot.llm_queue.drop(self.channel_id)
        self.player.stop()
        log.info("команда молчания — прервал ответ и остановил музыку")

    async def say(self, text: str) -> None:
        """Озвучить фразу и дождаться, пока она прозвучит."""
        try:
            await self.arbiter.speak(self.bot.tts.stream(text))
        except Exception:
            log.exception("не смог озвучить ответ: %r", text)
        finally:
            # Речь отзвучала — реплика отработала целиком, можно печатать отчёт.
            turn_report()

    # -------------------------------------------------------- пайплайн ----
    async def _on_transcript(self, transcript: Transcript) -> None:
        speaker = self.speaker_name(transcript.user_id)
        match = self.bot.wakeword.match(transcript.text)

        if match is None:
            log.debug("%s (мимо): %s", speaker, transcript.text)
            # Реплика не наша: отчёта не будет, замеры её stt держать незачем.
            turn_drop()
            return

        turn_note(speaker=speaker, heard=transcript.text)
        log.debug("обращение от %s (score=%.0f): %r", speaker, match.score, match.payload)

        with stage(log, "intent"):
            decision = await self.bot.router.route(
                match.payload,
                music_playing=self.player.current is not None,
            )

        if decision.intent is Intent.SILENCE:
            self.silence()
            turn_note(sent=f"команда {decision.intent.value}", reply="(молчу)")
            # say() не будет, а отчёт закрывается именно там — иначе реплика
            # осталась бы висеть открытой.
            turn_report()
            return

        if decision.intent is Intent.JOKE:
            # Мимо диалоговой очереди: анекдот берётся из ленты, а не
            # сочиняется, и ждать за чужим инференсом ему незачем.
            await self._handle_joke(decision)
            return

        if decision.intent is Intent.CHAT:
            await self.bot.llm_queue.submit(
                ChatTask(
                    # Момент конца фразы, а не текущее время: очередь
                    # расставляет приоритет по нему, и время, потраченное на
                    # STT и разбор намерения, не должно двигать человека в конец.
                    ended_at=transcript.ended_at,
                    channel_id=self.channel_id,
                    user_id=transcript.user_id,
                    speaker=speaker,
                    text=match.payload or transcript.text,
                    cid=current_cid.get(),
                )
            )
            return

        # Музыкальные команды идут мимо диалоговой очереди — они быстрые,
        # и ждать за чужим инференсом им незачем.
        await self._handle_music(decision, speaker)

    async def _handle_joke(self, decision: Decision) -> None:
        try:
            answer = await self.bot.jokes.tell()
        except Exception:
            log.exception("анекдот не рассказался")
            answer = "Анекдот сломался, посмотри логи."

        turn_note(sent=f"команда {decision.intent.value}", reply=answer)
        await self.say(answer)

    async def _handle_music(self, decision: Decision, speaker: str) -> None:
        try:
            if decision.intent is Intent.MUSIC_PLAY:
                answer = await self.player.play(decision.query, speaker)
            elif decision.intent is Intent.MUSIC_PAUSE:
                answer = self.player.pause()
            elif decision.intent is Intent.MUSIC_RESUME:
                answer = self.player.resume()
            elif decision.intent is Intent.MUSIC_SKIP:
                answer = self.player.skip()
            elif decision.intent is Intent.MUSIC_STOP:
                answer = self.player.stop()
            elif decision.intent is Intent.MUSIC_LOUDER:
                answer = self.player.louder()
            elif decision.intent is Intent.MUSIC_QUIETER:
                answer = self.player.quieter()
            elif decision.intent is Intent.MUSIC_VOLUME:
                answer = self.player.volume(decision.level)
            else:  # pragma: no cover — все ветки перечислены выше
                answer = "Не понял, что сделать с музыкой."
        except Exception:
            log.exception("музыкальная команда %s упала", decision.intent.value)
            answer = "С музыкой что-то не так, посмотри логи."

        # Музыка идёт мимо ЛЛМ — в отчёте показываем команду и что ответили.
        turn_note(sent=f"команда {decision.intent.value}", reply=answer)
        await self.say(answer)


class BashmakBot(discord.Client):
    def __init__(self, cfg: Config) -> None:
        intents = discord.Intents.default()
        intents.voice_states = True
        # Нужны, чтобы обращаться к людям по нику (privileged intent —
        # включается в настройках приложения на портале разработчика).
        intents.members = True

        super().__init__(intents=intents)
        self.cfg = cfg
        self.tree = app_commands.CommandTree(self)
        # Без этих двух «Ошибка взаимодействия» в Discord не оставляет в логе
        # ничего: interaction_check показывает, что команда до нас дошла,
        # on_error — что она упала (иначе исключение уходит в логгер
        # discord.app_commands, а он у нас прижат вместе со всем discord.*).
        self.tree.interaction_check = self._log_command
        self.tree.on_error = self._on_command_error

        self.stt = create_stt_pool(cfg.stt)
        self.tts = TtsPool(cfg.tts)
        self.llm = LlmClient(cfg.llm)
        self.wakeword = WakeWordFilter(cfg.wakeword)
        self.router = IntentRouter(cfg, self.llm)
        self.llm_queue = LlmQueue(cfg.llm, self.llm, self._on_llm_reply)
        # Один на весь бот: пачка из ленты и список рассказанного общие для
        # всех каналов, иначе в соседнем звучал бы тот же анекдот.
        self.jokes = JokeTeller(cfg, self.llm)

        self.sessions: dict[int, GuildSession] = {}
        self._idle_task: asyncio.Task | None = None
        self._levels_task: asyncio.Task | None = None

    # ---------------------------------------------------------- команды ----
    @staticmethod
    async def _log_command(interaction: discord.Interaction) -> bool:
        """Отметить, что команда до нас дошла. Ничего не проверяет, всегда True."""
        name = interaction.command.name if interaction.command else "?"
        log.info("команда /%s от %s", name, interaction.user)
        return True

    async def _on_command_error(
        self, interaction: discord.Interaction, error: Exception
    ) -> None:
        """Показать упавшую команду в логе и человеку, а не молча."""
        name = interaction.command.name if interaction.command else "?"
        log.error("команда /%s упала", name, exc_info=error)
        with contextlib.suppress(Exception):
            answer = "Команда упала — посмотри логи."
            if interaction.response.is_done():
                await interaction.followup.send(answer, ephemeral=True)
            else:
                await interaction.response.send_message(answer, ephemeral=True)

    # ------------------------------------------------------- жизненный цикл
    async def setup_hook(self) -> None:
        """Зарегистрировать slash-команды.

        py-cord синхронизировал дерево команд сам, discord.py — нет: без
        явного ``sync()`` команд в Discord просто не появится.
        """
        guild_ids = [int(g) for g in (self.cfg.discord.get("guild_ids") or [])]
        if not guild_ids:
            await self.tree.sync()
            log.warning(
                "guild_ids пуст — команды зарегистрированы глобально. Discord "
                "раскатывает их не мгновенно и держит на них общий лимит, а "
                "клиент до обновления может слать вызовы по устаревшему id — "
                "и тогда «Ошибка взаимодействия» не доедет до бота вовсе. "
                "Для отладки укажите id сервера в config.yaml (discord.guild_ids)."
            )
            return

        for guild_id in guild_ids:
            guild = discord.Object(id=guild_id)
            self.tree.copy_global_to(guild=guild)
            await self.tree.sync(guild=guild)
        log.info("команды зарегистрированы на серверах: %s", guild_ids)

    async def on_ready(self) -> None:
        log.info("вошёл как %s (id=%s)", self.user, self.user.id)

        if not await self.llm.wait_until_ready():
            log.error(
                "llama-server не поднялся на %s — диалог работать не будет. "
                "Проверьте: systemctl status llama-server",
                self.llm.base_url,
            )
        else:
            log.info("llama-server готов")

        self.llm_queue.start()
        if self._idle_task is None:
            self._idle_task = asyncio.create_task(self._idle_watch(), name="idle-watch")
        if self._levels_task is None:
            self._levels_task = asyncio.create_task(self._levels_watch(), name="levels")

    async def close(self) -> None:
        log.info("останавливаюсь...")
        for name in ("_idle_task", "_levels_task"):
            task = getattr(self, name)
            if task is None:
                continue
            task.cancel()
            # Дождаться отмены, иначе на выходе прилетит
            # «Task was destroyed but it is pending».
            with contextlib.suppress(asyncio.CancelledError):
                await task
            setattr(self, name, None)
        for session in list(self.sessions.values()):
            await session.close()
        self.sessions.clear()

        await self.llm_queue.stop()
        await self.jokes.close()
        await self.llm.close()
        self.stt.close()
        self.tts.close()
        music_search.shutdown()
        await super().close()
        log.info("остановлен")

    async def _on_llm_reply(self, task: ChatTask, reply: str) -> None:
        session = self._session_for_channel(task.channel_id)
        if session is None:
            # Пока модель думала, бота могли выгнать из канала.
            log.warning("некому озвучить ответ: сессия для канала %s закрыта", task.channel_id)
            return
        await session.say(reply)

    def _session_for_channel(self, channel_id: int) -> GuildSession | None:
        for session in self.sessions.values():
            if session.channel_id == channel_id:
                return session
        return None

    @guard("публикация уровней", reraise=False)
    async def _levels_watch(self) -> None:
        """Выкладывать уровни говорящих для монитора (bashmak/monitor.py).

        Отдельной задачей, а не из слушателя: в аудиотракте, который крутится
        каждые 30 мс, файловый ввод-вывод недопустим. Здесь он раз в полсекунды
        и мимо обработки речи.
        """
        path = levels_path(self.cfg)
        empty_published = False
        while True:
            await asyncio.sleep(LEVELS_INTERVAL)
            users: list[dict[str, Any]] = []
            threshold = 0.5
            for session in list(self.sessions.values()):
                threshold = session.listener.vad_threshold
                for user_id, level in session.listener.levels().items():
                    users.append({"name": session.speaker_name(user_id), **level})

            if not users and empty_published:
                # Никого нет и монитор об этом уже знает — не трогаем диск.
                continue
            publish_levels(path, {"at": time.time(), "threshold": threshold, "users": users})
            empty_published = not users

    @guard("сторож простоя", reraise=False)
    async def _idle_watch(self) -> None:
        """Выйти из канала, где давно никого нет, чтобы не жечь CPU впустую."""
        timeout = float(self.cfg.discord.get("idle_leave_seconds", 900))
        while True:
            await asyncio.sleep(IDLE_CHECK_INTERVAL)
            if timeout <= 0:
                continue
            # Тело цикла целиком под try: любая ошибка (например, канал отвалился
            # и стал None) не должна убивать сторожа до конца жизни процесса.
            try:
                await self._drop_empty_sessions(timeout)
            except Exception:
                log.exception("сторож простоя: не смог проверить каналы")

    async def _drop_empty_sessions(self, timeout: float) -> None:
        for guild_id, session in list(self.sessions.items()):
            channel = session.voice_client.channel
            humans = [m for m in getattr(channel, "members", []) if not m.bot]
            if humans:
                session.empty_since = None
                continue
            if session.empty_since is None:
                session.empty_since = time.monotonic()
            elif time.monotonic() - session.empty_since > timeout:
                log.info("в #%s никого — выхожу", getattr(channel, "name", "?"))
                self.sessions.pop(guild_id, None)
                await session.close()


async def _reject_dm(interaction: discord.Interaction) -> bool:
    """С пустым guild_ids команды регистрируются глобально и видны в личке.

    Там ``interaction.guild`` — None, и любое обращение к нему упало бы с
    AttributeError.
    """
    if interaction.guild is None:
        await interaction.response.send_message("Эта команда работает только на сервере.", ephemeral=True)
        return True
    return False


def build_bot(cfg: Config) -> BashmakBot:
    bot = BashmakBot(cfg)

    @bot.tree.command(name="start", description="Позвать Башмака в свой голосовой канал")
    async def start(interaction: discord.Interaction) -> None:
        if await _reject_dm(interaction):
            return

        voice = interaction.user.voice
        if voice is None or voice.channel is None:
            await interaction.response.send_message("Сначала зайди в голосовой канал.", ephemeral=True)
            return

        # Строго до закрытия старой сессии и подключения: и то и другое дольше
        # трёх секунд, которые Discord даёт на ответ, а просроченный ответ —
        # это «Ошибка взаимодействия» у пользователя и NotFound у нас.
        # После defer() отвечать можно только через followup.
        await interaction.response.defer(ephemeral=True)

        existing = bot.sessions.pop(interaction.guild.id, None)
        if existing is not None:
            await existing.close()
        voice_client: voice_recv.VoiceRecvClient | None = None
        try:
            voice_client = await voice.channel.connect(cls=voice_recv.VoiceRecvClient)
            session = GuildSession(bot, voice_client)
            await session.start()
            bot.sessions[interaction.guild.id] = session
        except Exception:
            log.exception("не смог зайти в голосовой канал")
            # Соединение могло уже подняться — без этого бот молча висел бы в
            # канале без сессии, и выгнать его было бы нечем: /leave не знает
            # о таком «призраке».
            if voice_client is not None:
                with contextlib.suppress(Exception):
                    await voice_client.disconnect(force=True)
            await interaction.followup.send("Не получилось подключиться — посмотри логи.", ephemeral=True)
            return

        await interaction.followup.send(
            f"Слушаю в #{voice.channel.name}. Зови по имени: «Башмак, ...»", ephemeral=True
        )

    @bot.tree.command(name="leave", description="Выгнать Башмака из голосового канала")
    async def leave(interaction: discord.Interaction) -> None:
        if await _reject_dm(interaction):
            return
        session = bot.sessions.pop(interaction.guild.id, None)
        if session is None:
            await interaction.response.send_message("Меня и так нигде нет.", ephemeral=True)
            return
        await session.close()
        await interaction.response.send_message("Вышел.", ephemeral=True)

    @bot.tree.command(name="status", description="Что сейчас происходит")
    async def status(interaction: discord.Interaction) -> None:
        if await _reject_dm(interaction):
            return
        session = bot.sessions.get(interaction.guild.id)
        lines = [
            f"llama-server: {'готов' if await bot.llm.health() else 'НЕ ОТВЕЧАЕТ'}",
            f"очередь к LLM: {bot.llm_queue.depth}",
        ]
        if session is None:
            lines.append("в голосовом канале: нет")
        else:
            track = session.player.current
            channel = session.voice_client.channel
            lines.append(f"канал: #{getattr(channel, 'name', '?')}")
            lines.append(f"говорящих в обработке: {len(session.listener.registry.snapshot())}")
            if track is None:
                lines.append("музыка: не играет")
            else:
                suffix = " (пауза)" if session.player.is_paused else ""
                lines.append(f"музыка: {track.title}{suffix}")
            if session.player.queued:
                lines.append(f"в очереди треков: {len(session.player.queued)}")
            lines.append(f"радио: {'включено' if session.player.radio_on else 'выключено'}")
        lines.append("")
        lines.extend(stage_summary())
        await interaction.response.send_message(
            "```\n" + "\n".join(lines) + "\n```", ephemeral=True
        )

    return bot


def main() -> None:
    cfg = load_config()
    setup_logging(cfg)
    log.info("Башмак стартует (конфиг: %s, профиль: %s)", cfg.source, cfg.get("profile", "?"))
    # До первого подключения к голосу: расшифровка DAVE и живучесть приёма
    # (см. bashmak/audio/voice_recv_patch.py).
    voice_recv_patch.apply()

    bot = build_bot(cfg)
    try:
        # log_handler=None — иначе run() настраивает логирование сам: вешает
        # свой хендлер (каждая строка библиотеки печатается дважды) и ставит
        # логгеру discord уровень INFO уже ПОСЛЕ нашего setup_logging().
        bot.run(cfg.discord_token, log_handler=None)
    except KeyboardInterrupt:  # pragma: no cover
        log.info("прервано с клавиатуры")


if __name__ == "__main__":
    main()
