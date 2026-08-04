"""Проигрыватель: очередь треков поверх микшера.

Плеер не трогает ``voice_client`` — он только подсовывает микшеру источник.
Всё, что связано с приоритетом речи над музыкой, живёт в output/arbiter.py.
"""

from __future__ import annotations

import logging
from collections import deque

import discord

from ..output.arbiter import OutputArbiter
from .search import SearchError, Track, search

log = logging.getLogger(__name__)

# -reconnect: стрим с YouTube регулярно рвётся на длинных треках,
# без этих флагов ffmpeg просто молча завершится в середине песни.
_FFMPEG_BEFORE = (
    "-nostdin -reconnect 1 -reconnect_streamed 1 -reconnect_delay_max 5 -loglevel warning"
)
_FFMPEG_OPTIONS = "-vn"

VOLUME_STEP = 0.15


class MusicPlayer:
    def __init__(self, cfg, arbiter: OutputArbiter) -> None:  # noqa: ANN001
        self._cfg = cfg.music
        self._arbiter = arbiter
        self._queue: deque[Track] = deque()
        self._current: Track | None = None
        # Источник текущего трека — «удостоверение личности» для _on_track_end:
        # колбэк приезжает через луп и легко может относиться к треку, который
        # мы уже сами сняли (см. skip).
        self._source: discord.AudioSource | None = None
        self._max_queue = int(self._cfg.get("max_queue", 20))
        self._search_timeout = float(self._cfg.get("search_timeout_s", 20))

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

    # -------------------------------------------------------- управление ---
    async def play(self, query: str, requested_by: str = "") -> str:
        """Найти и поставить трек. Возвращает фразу, которую бот скажет вслух."""
        if not query:
            return "А что включить-то? Скажи название."

        try:
            track = await search(query, self._search_timeout)
        except SearchError as exc:
            log.warning("поиск не удался: %s", exc)
            return "Не нашёл такого. Попробуй сказать по-другому."

        track.requested_by = requested_by

        if self._current is not None:
            if len(self._queue) >= self._max_queue:
                return "Очередь забита, подожди немного."
            self._queue.append(track)
            log.info("в очередь: %s (позиция %d)", track.title, len(self._queue))
            return f"Добавил в очередь: {track.title}"

        self._start(track)
        return f"Включаю: {track.title}"

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
        # set_music(None) не дёргает on_music_end, поэтому следующий трек
        # запускаем руками.
        self._release()
        if not self._advance():
            return f"Пропустил {skipped}. Очередь пуста."
        return f"Пропустил. Дальше: {self._current.title}"

    def stop(self) -> str:
        if self._current is None and not self._queue:
            return "И так тихо."
        self._queue.clear()
        self._release()
        return "Выключил музыку."

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
        self._queue.clear()
        self._release()

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
        log.info("играет: %s", track.title)

    def _advance(self) -> bool:
        if not self._queue:
            return False
        self._start(self._queue.popleft())
        return True

    def _on_track_end(self, source: discord.AudioSource) -> None:
        """Вызывается микшером (через луп), когда ffmpeg закончил трек."""
        if source is not self._source:
            # Трек кончился ровно в тот момент, когда его и так сняли (skip,
            # stop). Без этой проверки мы бы промотали ещё один трек и
            # рассинхронизировали _current с тем, что реально звучит.
            log.debug("трек закончился, но его уже сняли — колбэк устарел")
            return

        finished = self._current.title if self._current else "?"
        self._current = None
        self._source = None
        log.info("трек закончился: %s", finished)
        try:
            self._advance()
        except Exception:
            log.exception("не смог перейти к следующему треку")
