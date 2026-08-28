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
import re
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Container, Iterable, Sequence

import yt_dlp

log = logging.getLogger(__name__)

#: Свой пул, а не общий executor лупа: ``asyncio.wait_for`` снимает ожидание,
#: но не сам поток — yt-dlp висит на сети до своего socket_timeout. На общем
#: executor'е серия неудачных поисков заняла бы его целиком; здесь худший
#: случай локален — поиск ждёт свободного слота и возвращает «не нашёл».
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

#: Пол длительности: короче — это Shorts и обрезки, а не трек.
_MIN_RANDOM_DURATION = 70

#: Сколько секунд бюджета держать про запас на резолв выбранного ролика.
#:
#: Перебор источников останавливается за этот срок до конца таймаута: иначе
#: последний перечень успевал бы, а резолв — уже нет, и вызов возвращал бы
#: «не нашёл», имея на руках готовую ссылку.
_RESOLVE_RESERVE = 6.0

#: Чего в выдаче быть не должно. Проверяется по названию и по каналу:
#:
#: * бразильский фонк — по запросу «phonk» его больше, чем всего остального;
#: * ИИ-генерация — у неё свои опознавательные знаки в названии и канале;
#: * переливки чужих треков (nightcore, slowed, bass boosted) — на волне подряд
#:   они звучат одинаково.
#:
#: Слова подобраны узко, по устойчивым связкам: голое «ai» ловило бы японские
#: названия, голое «funk» — обычный фанк.
_BLOCKED = re.compile(
    r"brazil|brasil|бразил|baile|mandel[ãa]o|automotivo|montagem|\bmtg\b"
    r"|funk\s*(?:brasil|br\b)|\bai\s*(?:music|song|cover|generated|remix)"
    r"|generated\s+by\s+ai|\bsuno\b|\budio\b|нейросет|нейронк"
    r"|nightcore|bass\s*boost|slowed|sped\s*up|8d\s*audio",
    re.IGNORECASE,
)


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


def _choose_url(
    entries: Iterable[dict[str, Any]],
    exclude: Container[str] = (),
    min_views: int = 0,
) -> str:
    """Случайная годная ссылка из выдачи. Пустая строка — годных не осталось.

    Отсеиваются прямые эфиры (у них нет конца, и волна на них зависает),
    многочасовые сборники, Shorts, мусор из :data:`_BLOCKED` и то, что недавно
    уже играло.

    ``min_views`` — главный фильтр качества: ноунеймы и ИИ-генерация отличаются
    от нормальной музыки не словами в названии, а порядком просмотров.
    """
    good: list[str] = []
    # Записи без счётчика просмотров — второй эшелон, а не мусор: поле в плоской
    # выдаче необязательное, и жёсткий отказ выключил бы волну целиком, стоит
    # YouTube перестать его отдавать.
    unknown: list[str] = []

    for entry in entries:
        if not entry or entry.get("is_live") or entry.get("live_status") == "is_live":
            continue
        duration = entry.get("duration")
        if duration is not None and not (_MIN_RANDOM_DURATION <= duration <= _MAX_RANDOM_DURATION):
            continue
        if _BLOCKED.search(f"{entry.get('title') or ''} {entry.get('channel') or entry.get('uploader') or ''}"):
            continue
        url = _entry_url(entry)
        if not url or url in exclude:
            continue

        views = entry.get("view_count")
        if not min_views or (views is not None and views >= min_views):
            good.append(url)
        elif views is None:
            unknown.append(url)

    pool = good or unknown
    return random.choice(pool) if pool else ""


def _extract_random(
    sources: list[str], pool: int, exclude: Container[str], min_views: int, deadline: float
) -> Track:
    """Блокирующая часть случайного выбора: перечень выдачи, затем один резолв.

    Источники перебираются в случайном порядке до первого удачного: после
    фильтров выдача часто пустеет, и одной попытки мало.

    ``deadline`` (``time.monotonic``) — до каких пор есть смысл искать. Перебор
    ограничен временем, а не числом попыток: ``asyncio.wait_for`` в
    :func:`search_random` снимает ожидание, но не этот поток, и без бюджета он
    ходил бы по оставшимся источникам, когда вызвавшего уже нет. Пул на два
    потока, и пара таких заданий занимает его целиком.
    """
    tried = random.sample(sources, k=len(sources))
    used = 0
    for source in tried:
        # Первый источник пробуем всегда: без него вызов заведомо пустой. Дальше
        # — пока в бюджете остаётся время ещё и на резолв найденного.
        if used and time.monotonic() + _RESOLVE_RESERVE >= deadline:
            log.debug("случайный трек: время вышло, перебрано источников %d из %d", used, len(tried))
            break
        used += 1

        with yt_dlp.YoutubeDL(_FLAT_OPTIONS) as ydl:
            info = ydl.extract_info(_source_query(source, pool), download=False)

        url = _choose_url((info or {}).get("entries") or [], exclude, min_views)
        if url:
            log.debug("случайный трек взят из источника %r", source)
            return _extract(url)
        log.debug("источник %r не дал ничего годного", source)

    raise SearchError(f"ни один из источников не дал трека: {tried[:used]}")


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
    min_views: int = 0,
) -> Track:
    """Выбрать трек за человека: случайный источник, случайный трек из выдачи.

    ``sources`` — строки-запросы и ссылки на плейлисты (music.random_sources).
    ``exclude`` — страницы недавно игравших треков, чтобы волна не крутила одно
    и то же. ``min_views`` — порог популярности (music.random_min_views).

    Оба похода в сеть (перечень и резолв выбранного) идут одним заданием в
    тредпуле, поэтому таймаут здесь общий на них — как и у :func:`search`.
    """
    if not sources:
        raise SearchError("список источников пуст (music.random_sources)")

    loop = asyncio.get_running_loop()
    # Отсчёт идёт от постановки в очередь, а не от старта задания: ожидание
    # свободного слота тратит тот же таймаут, и задание, дождавшееся его слишком
    # поздно, лезть в сеть уже не должно.
    deadline = time.monotonic() + timeout
    try:
        track = await asyncio.wait_for(
            loop.run_in_executor(
                _executor(),
                _extract_random,
                list(sources),
                pool,
                frozenset(exclude),
                min_views,
                deadline,
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
