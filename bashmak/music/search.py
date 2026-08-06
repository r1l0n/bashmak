"""Поиск трека и получение прямой ссылки на аудио-поток.

На диск ничего не скачивается: yt-dlp отдаёт URL потока, ffmpeg читает его
напрямую. Быстрее старт и не растёт занятое место на сервере.
"""

from __future__ import annotations

import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import yt_dlp

log = logging.getLogger(__name__)

#: Свой пул, а не общий executor лупа.
#:
#: ``asyncio.wait_for`` снимает ожидание, но не сам поток: yt-dlp продолжает
#: висеть на сети до своего socket_timeout. На executor'е по умолчанию серия
#: неудачных поисков заняла бы его целиком вместе со всеми остальными
#: run_in_executor в процессе. С отдельным пулом худший случай локален: поиск
#: дожидается свободного слота и возвращает «не нашёл».
_EXECUTOR: ThreadPoolExecutor | None = None

_YDL_OPTIONS = {
    "format": "bestaudio/best",
    "noplaylist": True,
    "quiet": True,
    "no_warnings": True,
    "default_search": "ytsearch1",
    "source_address": "0.0.0.0",  # обходит залипание на IPv6 у некоторых хостеров
    "extract_flat": False,
    "socket_timeout": 15,
}


@dataclass(slots=True)
class Track:
    title: str
    stream_url: str
    page_url: str
    duration: float | None
    requested_by: str = ""


class SearchError(RuntimeError):
    pass


def _extract(query: str) -> Track:
    """Блокирующая часть: сеть и разбор. Выполняется в тредпуле."""
    with yt_dlp.YoutubeDL(_YDL_OPTIONS) as ydl:
        info = ydl.extract_info(query, download=False)

    if info is None:
        raise SearchError(f"ничего не нашлось по запросу {query!r}")

    # Поисковый запрос возвращает плейлист из одного элемента.
    if "entries" in info:
        entries = [entry for entry in info["entries"] if entry]
        if not entries:
            raise SearchError(f"ничего не нашлось по запросу {query!r}")
        info = entries[0]

    stream_url = info.get("url")
    if not stream_url:
        raise SearchError(f"у найденного трека нет аудио-потока: {info.get('title')!r}")

    return Track(
        title=info.get("title") or "без названия",
        stream_url=stream_url,
        page_url=info.get("webpage_url") or "",
        duration=info.get("duration"),
    )


def _executor() -> ThreadPoolExecutor:
    global _EXECUTOR
    if _EXECUTOR is None:
        _EXECUTOR = ThreadPoolExecutor(max_workers=2, thread_name_prefix="ytdlp")
    return _EXECUTOR


def shutdown() -> None:
    """Закрыть пул поиска. Зависшие потоки завершатся по таймауту сокета."""
    global _EXECUTOR
    if _EXECUTOR is not None:
        _EXECUTOR.shutdown(wait=False, cancel_futures=True)
        _EXECUTOR = None


async def search(query: str, timeout: float = 20.0) -> Track:
    """Найти трек по названию или ссылке."""
    loop = asyncio.get_running_loop()
    try:
        track = await asyncio.wait_for(loop.run_in_executor(_executor(), _extract, query), timeout)
    except asyncio.TimeoutError as exc:
        raise SearchError(f"поиск {query!r} занял больше {timeout:.0f} с") from exc
    except SearchError:
        raise
    except Exception as exc:  # у yt-dlp свои типы ошибок почти на каждый случай
        raise SearchError(f"не смог найти {query!r}: {exc}") from exc

    log.debug("найден трек: %s (%s)", track.title, track.page_url)
    return track
