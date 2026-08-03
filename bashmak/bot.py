"""Точка входа: py-cord клиент и сборка всего пайплайна.

Что здесь важно понимать про распределение ресурсов:

* приём (VAD + STT) параллелен по говорящим — у каждого свой буфер, свой VAD,
  и сегменты уходят в пул процессов независимо;
* LLM одна на всех и обслуживается строго по одному запросу — это узкое место
  CPU, и очередь тут не костыль, а способ не терять реплики;
* голосовой выход тоже один — им рулит OutputArbiter.

Соответственно STT/TTS-пулы, клиент LLM и очередь общие на процесс, а
слушатель, арбитр и плеер — свои на каждый сервер (guild).
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import time

import discord

from .audio.listener import VoiceListener
from .config import Config, load_config
from .intent.router import Decision, Intent, IntentRouter
from .llm.client import LlmClient
from .llm.queue_manager import ChatTask, LlmQueue
from .music.player import MusicPlayer
from .output.arbiter import OutputArbiter
from .stt.whisper_worker import SttPool, Transcript
from .tts.piper_worker import TtsPool
from .utils.logging import current_cid, guard, setup_logging, stage
from .wakeword.filter import WakeWordFilter

log = logging.getLogger(__name__)

IDLE_CHECK_INTERVAL = 30.0


class GuildSession:
    """Всё, что живёт ровно столько, сколько бот сидит в голосовом канале."""

    def __init__(self, bot: "BashmakBot", voice_client: discord.VoiceClient) -> None:
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
        self.voice_client.start_recording(self.listener.sink, self._on_recording_stop)
        self.listener.start()
        log.info("сессия запущена в #%s (%s)", self.voice_client.channel.name, self.guild.name)

    async def close(self) -> None:
        self.arbiter.interrupt()
        self.player.shutdown()
        await self.listener.stop()

        with contextlib.suppress(Exception):
            if self.voice_client.recording:
                self.voice_client.stop_recording()
        with contextlib.suppress(Exception):
            self.voice_client.stop()
        with contextlib.suppress(Exception):
            await self.voice_client.disconnect(force=True)

        self.bot.llm_queue.reset(self.channel_id)
        log.info("сессия в %s закрыта", self.guild.name)

    async def _on_recording_stop(self, sink, *args) -> None:  # noqa: ANN001 — сигнатура py-cord
        log.debug("запись остановлена")

    # ------------------------------------------------------------ речь ----
    def speaker_name(self, user_id: int) -> str:
        member = self.guild.get_member(user_id) if self.guild else None
        if member is not None:
            return member.display_name
        user = self.bot.get_user(user_id)
        return user.display_name if user is not None else f"user{user_id}"

    async def say(self, text: str) -> None:
        """Озвучить фразу и дождаться, пока она прозвучит."""
        try:
            await self.arbiter.speak(self.bot.tts.stream(text))
        except Exception:
            log.exception("не смог озвучить ответ: %r", text)

    # -------------------------------------------------------- пайплайн ----
    async def _on_transcript(self, transcript: Transcript) -> None:
        speaker = self.speaker_name(transcript.user_id)
        match = self.bot.wakeword.match(transcript.text)

        if match is None:
            # Реплика не боту, но пусть останется контекстом разговора.
            self.bot.llm_queue.note_user_line(self.channel_id, speaker, transcript.text)
            log.debug("%s (мимо): %s", speaker, transcript.text)
            return

        log.info("обращение от %s (score=%.0f): %r", speaker, match.score, match.payload)

        with stage(log, "intent"):
            decision = await self.bot.router.route(match.payload)

        if decision.intent is Intent.CHAT:
            await self.bot.llm_queue.submit(
                ChatTask(
                    ended_at=time.monotonic(),
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
            else:  # pragma: no cover — все ветки перечислены выше
                answer = "Не понял, что сделать с музыкой."
        except Exception:
            log.exception("музыкальная команда %s упала", decision.intent.value)
            answer = "С музыкой что-то не так, посмотри логи."

        await self.say(answer)


class BashmakBot(discord.Bot):
    def __init__(self, cfg: Config) -> None:
        intents = discord.Intents.default()
        intents.voice_states = True
        # Нужны, чтобы обращаться к людям по нику (privileged intent —
        # включается в настройках приложения на портале разработчика).
        intents.members = True

        super().__init__(intents=intents)
        self.cfg = cfg

        self.stt = SttPool(cfg.stt)
        self.tts = TtsPool(cfg.tts)
        self.llm = LlmClient(cfg.llm)
        self.wakeword = WakeWordFilter(cfg.wakeword)
        self.router = IntentRouter(cfg, self.llm)
        self.llm_queue = LlmQueue(cfg.llm, self.llm, self._on_llm_reply)

        self.sessions: dict[int, GuildSession] = {}
        self._idle_task: asyncio.Task | None = None

    # ------------------------------------------------------- жизненный цикл
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

    async def close(self) -> None:
        log.info("останавливаюсь...")
        if self._idle_task is not None:
            self._idle_task.cancel()
        for session in list(self.sessions.values()):
            await session.close()
        self.sessions.clear()

        await self.llm_queue.stop()
        await self.llm.close()
        self.stt.close()
        self.tts.close()
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

    @guard("сторож простоя")
    async def _idle_watch(self) -> None:
        """Выйти из канала, где давно никого нет, чтобы не жечь CPU впустую."""
        timeout = float(self.cfg.discord.get("idle_leave_seconds", 900))
        while True:
            await asyncio.sleep(IDLE_CHECK_INTERVAL)
            if timeout <= 0:
                continue
            for guild_id, session in list(self.sessions.items()):
                channel = session.voice_client.channel
                humans = [m for m in getattr(channel, "members", []) if not m.bot]
                if humans:
                    session.empty_since = None
                    continue
                if session.empty_since is None:
                    session.empty_since = time.monotonic()
                elif time.monotonic() - session.empty_since > timeout:
                    log.info("в #%s никого — выхожу", channel.name)
                    await session.close()
                    self.sessions.pop(guild_id, None)


def build_bot(cfg: Config) -> BashmakBot:
    bot = BashmakBot(cfg)
    guild_ids = list(cfg.discord.get("guild_ids", []) or []) or None

    @bot.slash_command(name="join", description="Позвать Башмака в свой голосовой канал", guild_ids=guild_ids)
    async def join(ctx: discord.ApplicationContext) -> None:
        voice = ctx.author.voice
        if voice is None or voice.channel is None:
            await ctx.respond("Сначала зайди в голосовой канал.", ephemeral=True)
            return

        existing = bot.sessions.pop(ctx.guild.id, None)
        if existing is not None:
            await existing.close()

        await ctx.defer(ephemeral=True)
        try:
            voice_client = await voice.channel.connect()
            session = GuildSession(bot, voice_client)
            await session.start()
            bot.sessions[ctx.guild.id] = session
        except Exception:
            log.exception("не смог зайти в голосовой канал")
            await ctx.respond("Не получилось подключиться — посмотри логи.", ephemeral=True)
            return

        await ctx.respond(f"Слушаю в #{voice.channel.name}. Зови по имени: «Башмак, ...»", ephemeral=True)

    @bot.slash_command(name="leave", description="Выгнать Башмака из голосового канала", guild_ids=guild_ids)
    async def leave(ctx: discord.ApplicationContext) -> None:
        session = bot.sessions.pop(ctx.guild.id, None)
        if session is None:
            await ctx.respond("Меня и так нигде нет.", ephemeral=True)
            return
        await session.close()
        await ctx.respond("Вышел.", ephemeral=True)

    @bot.slash_command(name="status", description="Что сейчас происходит", guild_ids=guild_ids)
    async def status(ctx: discord.ApplicationContext) -> None:
        session = bot.sessions.get(ctx.guild.id)
        lines = [
            f"llama-server: {'готов' if await bot.llm.health() else 'НЕ ОТВЕЧАЕТ'}",
            f"очередь к LLM: {bot.llm_queue.depth}",
        ]
        if session is None:
            lines.append("в голосовом канале: нет")
        else:
            track = session.player.current
            lines.append(f"канал: #{session.voice_client.channel.name}")
            lines.append(f"говорящих в обработке: {len(session.listener.registry.snapshot())}")
            if track is None:
                lines.append("музыка: не играет")
            else:
                suffix = " (пауза)" if session.player.is_paused else ""
                lines.append(f"музыка: {track.title}{suffix}")
            if session.player.queued:
                lines.append(f"в очереди треков: {len(session.player.queued)}")
        await ctx.respond("\n".join(lines), ephemeral=True)

    @bot.slash_command(name="forget", description="Забыть историю разговора", guild_ids=guild_ids)
    async def forget(ctx: discord.ApplicationContext) -> None:
        session = bot.sessions.get(ctx.guild.id)
        if session is not None:
            bot.llm_queue.reset(session.channel_id)
        await ctx.respond("Забыл, о чём говорили.", ephemeral=True)

    return bot


def main() -> None:
    cfg = load_config()
    setup_logging(cfg)
    log.info("Башмак стартует (конфиг: %s, профиль: %s)", cfg.source, cfg.get("profile", "?"))

    bot = build_bot(cfg)
    try:
        bot.run(cfg.discord_token)
    except KeyboardInterrupt:  # pragma: no cover
        log.info("прервано с клавиатуры")


if __name__ == "__main__":
    main()
