"""Поиск трека и получение прямой ссылки на аудио-поток.

На диск ничего не скачивается: yt-dlp отдаёт URL потока, ffmpeg читает его
напрямую. Быстрее старт и не растёт занятое место на сервере.

Два входа: :func:`search` — по названию от человека, :func:`search_random` —
«включи что-нибудь», когда выбирать приходится за него.
"""

from __future__ import annotations

import asyncio
import logging
import random
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Container, Iterable, Sequence

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

#: Перечень выдачи без резолва каждого ролика. Для случайного выбора нужны
#: только ссылка и длительность, а полное извлечение двадцати записей — это
#: двадцать походов в сеть вместо одного.
_FLAT_OPTIONS = {**_YDL_OPTIONS, "extract_flat": "in_playlist"}

#: Потолок длительности случайного трека. В выдаче по жанру половина верхних
#: строк — часовые сборники и «10 hours»; радио на таком встаёт до вечера.
_MAX_RANDOM_DURATION = 900

#: Сколько источников перебрать, прежде чем сдаться. Выдача бывает пустой
#: (опечатка в конфиге) или целиком отсеянной фильтрами выше.
_SOURCE_TRIES = 3


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


def _source_query(source: str, pool: int) -> str:
    """Ссылка на плейлист уходит как есть, обычная строка — как поиск на pool штук."""
    if source.startswith(("http://", "https://")):
        return source
    return f"ytsearch{pool}:{source}"


def _entry_url(entry: dict[str, Any]) -> str:
    """Ссылка на ролик из плоской записи.

    В плоском режиме yt-dlp обычно кладёт в ``url`` готовую ссылку, но не
    всегда: часть экстракторов отдаёт только ``id``.
    """
    url = str(entry.get("url") or "")
    if url.startswith("http"):
        return url
    video_id = str(entry.get("id") or "")
    return f"https://www.youtube.com/watch?v={video_id}" if video_id else ""


def _choose_url(entries: Iterable[dict[str, Any]], exclude: Container[str] = ()) -> str:
    """Случайная годная ссылка из выдачи. Пустая строка — годных не осталось.

    Отсеиваются прямые эфиры (у них нет конца, и радио на них зависает),
    многочасовые сборники и то, что недавно уже играло.
    """
    pool: list[str] = []
    for entry in entries:
        if not entry or entry.get("is_live") or entry.get("live_status") == "is_live":
            continue
        duration = entry.get("duration")
        if duration is not None and duration > _MAX_RANDOM_DURATION:
            continue
        url = _entry_url(entry)
        if url and url not in exclude:
            pool.append(url)
    return random.choice(pool) if pool else ""


def _extract_random(sources: list[str], pool: int, exclude: Container[str]) -> Track:
    """Блокирующая часть случайного выбора: перечень выдачи, затем один резолв."""
    tried = random.sample(sources, k=min(len(sources), _SOURCE_TRIES))
    for source in tried:
        with yt_dlp.YoutubeDL(_FLAT_OPTIONS) as ydl:
            info = ydl.extract_info(_source_query(source, pool), download=False)

        url = _choose_url((info or {}).get("entries") or [], exclude)
        if url:
            log.debug("случайный трек взят из источника %r", source)
            return _extract(url)
        log.debug("источник %r не дал ничего годного", source)

    raise SearchError(f"ни один из источников не дал трека: {tried}")


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


async def search_random(
    sources: Sequence[str],
    pool: int = 20,
    timeout: float = 20.0,
    exclude: Container[str] = (),
) -> Track:
    """Выбрать трек за человека: случайный источник, случайный трек из выдачи.

    ``sources`` — строки-запросы и ссылки на плейлисты (music.random_sources).
    ``exclude`` — страницы недавно игравших треков, чтобы радио не крутило одно
    и то же.

    Оба похода в сеть (перечень и резолв выбранного) идут одним заданием в
    тредпуле, поэтому таймаут здесь общий на них — как и у :func:`search`.
    """
    if not sources:
        raise SearchError("список источников пуст (music.random_sources)")

    loop = asyncio.get_running_loop()
    try:
        track = await asyncio.wait_for(
            loop.run_in_executor(
                _executor(), _extract_random, list(sources), pool, frozenset(exclude)
            ),
            timeout,
        )
    except asyncio.TimeoutError as exc:
        raise SearchError(f"случайный трек не выбрался за {timeout:.0f} с") from exc
    except SearchError:
        raise
    except Exception as exc:  # у yt-dlp свои типы ошибок почти на каждый случай
        raise SearchError(f"не смог выбрать трек: {exc}") from exc

    log.debug("выбран случайный трек: %s (%s)", track.title, track.page_url)
    return track
