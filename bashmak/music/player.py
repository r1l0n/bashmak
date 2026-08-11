"""Проигрыватель: очередь треков поверх микшера.

Плеер не трогает ``voice_client`` — он только подсовывает микшеру источник.
Всё, что связано с приоритетом речи над музыкой, живёт в output/arbiter.py.

Названия трека может и не быть: «включи что-нибудь» — команда выбрать самому.
Тогда трек берётся из music.random_sources (см. search.search_random), и этим же
механизмом плеер продолжает играть, когда очередь опустела, — радио.
"""

from __future__ import annotations

import asyncio
import logging
from collections import deque

import discord

from ..output.arbiter import OutputArbiter
from ..utils.logging import guard
from .search import SearchError, Track, search, search_random

log = logging.getLogger(__name__)

# -reconnect: стрим с YouTube регулярно рвётся на длинных треках,
# без этих флагов ffmpeg просто молча завершится в середине песни.
_FFMPEG_BEFORE = (
    "-nostdin -reconnect 1 -reconnect_streamed 1 -reconnect_delay_max 5 -loglevel warning"
)
_FFMPEG_OPTIONS = "-vn"

VOLUME_STEP = 0.15

#: Сколько последних треков помнить, чтобы случайный выбор их не повторял.
#: Больше запаса не нужно: источников в конфиге единицы, и слишком длинная
#: память просто выест из них все годные варианты.
RECENT_LIMIT = 30


class MusicPlayer:
    def __init__(self, cfg, arbiter: OutputArbiter) -> None:  # noqa: ANN001
        self._cfg = cfg.music
        self._arbiter = arbiter
        self._queue: deque[Track] = deque()
        self._current: Track | None = None
        # Источник текущего трека — идентификатор для _on_track_end: колбэк
        # приходит через луп и может относиться к треку, который уже сняли
        # (см. skip).
        self._source: discord.AudioSource | None = None
        self._max_queue = int(self._cfg.get("max_queue", 20))
        self._search_timeout = float(self._cfg.get("search_timeout_s", 20))

        # Откуда брать трек, когда название не назвали.
        self._sources = [str(s) for s in (self._cfg.get("random_sources") or [])]
        self._pool = int(self._cfg.get("random_pool", 20))
        self._autoplay = bool(self._cfg.get("autoplay", True))
        #: Страницы недавно игравших треков — исключаются из случайного выбора.
        self._recent: deque[str] = deque(maxlen=RECENT_LIMIT)
        #: Радио взводится любой командой play и снимается «выключи музыку».
        #: Само по себе, без команды, оно не заводится: бот, начинающий играть
        #: в тишине, — это не радио, а сюрприз.
        self._radio_armed = False
        self._radio_task: asyncio.Task | None = None

        arbiter.bind_music_end(self._on_track_end)

    # ----------------------------------------------------------- статус ---
    @property
    def current(self) -> Track | None:
        return self._current

    @property
    def queued(self) -> list[Track]:
        return list(self._queue)

    @property
    def is_paused(self) -> bool:
        return self._arbiter.source.music_paused

    @property
    def radio_on(self) -> bool:
        """Продолжит ли плеер сам, когда очередь опустеет."""
        return self._autoplay and self._radio_armed and bool(self._sources)

    # -------------------------------------------------------- управление ---
    async def play(self, query: str, requested_by: str = "") -> str:
        """Найти и поставить трек. Возвращает фразу, которую бот скажет вслух.

        Пустой запрос — это «включи что-нибудь»: выбираем сами.
        """
        if not query:
            return await self._play_random(requested_by)

        try:
            track = await search(query, self._search_timeout)
        except SearchError as exc:
            log.warning("поиск не удался: %s", exc)
            return "Не нашёл такого. Попробуй сказать по-другому."

        return self._accept(track, requested_by, "Включаю")

    async def _play_random(self, requested_by: str) -> str:
        """«Включи что-нибудь»: трек из music.random_sources."""
        if not self._sources:
            return "А что включить-то? Скажи название."

        try:
            track = await search_random(
                self._sources, self._pool, self._search_timeout, self._recent
            )
        except SearchError as exc:
            log.warning("случайный трек не выбрался: %s", exc)
            return "Не нашёл, чего бы поставить. Скажи название."

        return self._accept(track, requested_by, "Сам выбрал")

    def _accept(self, track: Track, requested_by: str, verb: str) -> str:
        """Общий хвост обоих путей: в очередь или сразу в эфир."""
        track.requested_by = requested_by
        # Радио взводится любой командой play — дальше плеер продолжит сам.
        self._radio_armed = True

        if self._current is not None:
            if len(self._queue) >= self._max_queue:
                return "Очередь забита, подожди немного."
            self._queue.append(track)
            log.info("в очередь: %s (позиция %d)", track.title, len(self._queue))
            return f"Добавил в очередь: {track.title}"

        self._start(track)
        return f"{verb}: {track.title}"

    def pause(self) -> str:
        if self._current is None:
            return "Сейчас ничего не играет."
        self._arbiter.source.pause_music()
        return "Поставил на паузу."

    def resume(self) -> str:
        if self._current is None:
            return "Ставить с паузы нечего."
        self._arbiter.source.resume_music()
        return "Продолжаю."

    def skip(self) -> str:
        if self._current is None:
            return "Пропускать нечего."
        skipped = self._current.title
        # set_music(None) не вызывает on_music_end, поэтому следующий трек
        # запускается явно.
        self._release()
        if self._advance():
            return f"Пропустил. Дальше: {self._current.title}"
        if self._schedule_radio():
            return f"Пропустил {skipped}. Ищу следующий."
        return f"Пропустил {skipped}. Очередь пуста."

    def stop(self) -> str:
        was_playing = self._current is not None or bool(self._queue)
        # Радио снимается до всего остального: «выключи музыку» между треками
        # застаёт плеер за поиском следующего, и без отмены он бы всё равно
        # заиграл — через полминуты и как бы сам по себе.
        self._radio_off()
        self._queue.clear()
        self._release()
        return "Выключил музыку." if was_playing else "И так тихо."

    def louder(self) -> str:
        value = self._arbiter.source.set_volume(self._arbiter.source.volume + VOLUME_STEP)
        return f"Громкость {round(value * 100)} процентов."

    def quieter(self) -> str:
        value = self._arbiter.source.set_volume(self._arbiter.source.volume - VOLUME_STEP)
        return f"Громкость {round(value * 100)} процентов."

    def volume(self, level: int | None) -> str:
        """Выставить громкость в процентах: «громкость 70»."""
        if level is None:
            return "Какую громкость поставить? Скажи число."
        value = self._arbiter.source.set_volume(level / 100)
        return f"Громкость {round(value * 100)} процентов."

    def shutdown(self) -> None:
        self._radio_off()
        self._queue.clear()
        self._release()

    # ------------------------------------------------------------ радио ---
    def _radio_off(self) -> None:
        """Снять радио и отменить поиск, если он ещё идёт."""
        self._radio_armed = False
        if self._radio_task is not None:
            self._radio_task.cancel()
            self._radio_task = None

    def _schedule_radio(self) -> bool:
        """Завести фоновый поиск следующего трека. False — радио не встало.

        Только из лупа: конец трека приходит сюда через
        ``loop.call_soon_threadsafe`` (см. output/arbiter.py), команды — из
        обработки реплики.
        """
        if not self.radio_on:
            return False
        if self._radio_task is not None and not self._radio_task.done():
            return True
        try:
            self._radio_task = asyncio.create_task(self._radio_next(), name="radio")
        except RuntimeError:
            # Лупа уже нет — это выключение бота, продолжать нечего.
            return False
        return True

    @guard("радио", reraise=False)
    async def _radio_next(self) -> None:
        """Подобрать и запустить следующий трек. Вслух ничего не говорит.

        Объявлять каждый трек — это бот, заговаривающий сам с собой раз в три
        минуты. Что играет, видно в /status и в логе.

        Под guard: таск никто не ждёт, и без него любая неожиданная ошибка
        всплыла бы «Task exception was never retrieved» уже при сборке мусора.
        """
        try:
            track = await search_random(
                self._sources, self._pool, self._search_timeout, self._recent
            )
        except SearchError as exc:
            # Не долбимся в цикле: следующая команда play взведёт радио заново.
            log.warning("радио: следующий трек не нашёлся (%s), останавливаюсь", exc)
            self._radio_armed = False
            return

        if not self._radio_armed or self._current is not None:
            # Пока шёл поиск, человек успел сказать «включи Кино» или
            # «выключи музыку». Его выбор важнее нашего.
            log.debug("радио: пока искал, играет уже другое — трек не нужен")
            return
        self._start(track)

    # ---------------------------------------------------------- внутрь ----
    def _release(self) -> None:
        """Снять текущий трек, не запуская следующий."""
        self._current = None
        self._source = None
        self._arbiter.source.set_music(None)

    def _start(self, track: Track) -> None:
        source = discord.FFmpegPCMAudio(
            track.stream_url,
            before_options=_FFMPEG_BEFORE,
            options=_FFMPEG_OPTIONS,
        )
        self._current = track
        self._source = source
        self._arbiter.source.set_music(source)
        if track.page_url:
            self._recent.append(track.page_url)
        log.info("играет: %s", track.title)

    def _advance(self) -> bool:
        if not self._queue:
            return False
        self._start(self._queue.popleft())
        return True

    def _on_track_end(self, source: discord.AudioSource) -> None:
        """Вызывается микшером (через луп), когда ffmpeg закончил трек."""
        if source is not self._source:
            # Трек закончился ровно в тот момент, когда его уже сняли (skip,
            # stop). Без этой проверки промотался бы ещё один трек, и _current
            # разошёлся бы с тем, что реально звучит.
            log.debug("трек закончился, но его уже сняли: колбэк устарел")
            return

        finished = self._current.title if self._current else "?"
        self._current = None
        self._source = None
        log.info("трек закончился: %s", finished)
        try:
            if not self._advance():
                self._schedule_radio()
        except Exception:
            log.exception("не смог перейти к следующему треку")
