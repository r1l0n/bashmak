"""Логирование.

ТЗ требует, чтобы по логам было видно, на каком именно шаге затык. Поэтому:

* каждой реплике присваивается correlation id (`cid`) вида ``u123456@41.2``,
  который контекстной переменной протаскивается через все стадии обработки —
  sink → vad → stt → wake → intent → llm → tts → out;
* каждая стадия оборачивается в :func:`stage` и пишет свою длительность;
* фоновые задачи оборачиваются в :func:`guard`, чтобы упавший таск не умирал
  молча (asyncio по умолчанию проглатывает исключения в задачах).
"""

from __future__ import annotations

import asyncio
import contextvars
import functools
import logging
import logging.handlers
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Iterator

#: Идентификатор обрабатываемой сейчас реплики. Пустой — вне пайплайна.
current_cid: contextvars.ContextVar[str] = contextvars.ContextVar("cid", default="-")

_LOG_FORMAT = "%(asctime)s %(levelname)-7s [%(cid)s] %(name)s: %(message)s"
_DATE_FORMAT = "%H:%M:%S"


class _CidFilter(logging.Filter):
    """Подставляет cid в каждую запись, даже если её сделала чужая библиотека."""

    def filter(self, record: logging.LogRecord) -> bool:
        if not hasattr(record, "cid"):
            record.cid = current_cid.get()
        return True


def new_cid(user_id: int | str) -> str:
    """Свежий id реплики: кто говорит + момент времени (для сортировки в логе)."""
    return f"u{user_id}@{time.monotonic() % 10000:.1f}"


def setup_logging(cfg: Any = None) -> None:
    """Настроить корневой логгер: консоль (journald) + ротация в файл."""
    level_name = "INFO"
    log_file: Path | None = None
    max_bytes, backups = 10 * 1024 * 1024, 5

    if cfg is not None:
        section = cfg.get("logging") if hasattr(cfg, "get") else None
        if section is not None:
            level_name = str(section.get("level", level_name)).upper()
            max_bytes = int(section.get("max_bytes", max_bytes))
            backups = int(section.get("backup_count", backups))
            if "file" in section:
                log_file = section.path("file")

    level = getattr(logging, level_name, logging.INFO)
    formatter = logging.Formatter(_LOG_FORMAT, datefmt=_DATE_FORMAT)
    cid_filter = _CidFilter()

    root = logging.getLogger()
    root.setLevel(level)
    for existing in list(root.handlers):
        root.removeHandler(existing)

    console = logging.StreamHandler()
    console.setFormatter(formatter)
    console.addFilter(cid_filter)
    root.addHandler(console)

    if log_file is not None:
        try:
            log_file.parent.mkdir(parents=True, exist_ok=True)
            file_handler = logging.handlers.RotatingFileHandler(
                log_file, maxBytes=max_bytes, backupCount=backups, encoding="utf-8"
            )
            file_handler.setFormatter(formatter)
            file_handler.addFilter(cid_filter)
            root.addHandler(file_handler)
        except OSError as exc:
            root.warning("не удалось открыть лог-файл %s: %s (пишу только в консоль)", log_file, exc)

    # Эти двое очень болтливы на DEBUG и топят полезное.
    logging.getLogger("discord").setLevel(max(level, logging.WARNING))
    logging.getLogger("httpx").setLevel(max(level, logging.WARNING))

    # А эти двое болтливы на INFO, то есть даже когда discord поднимают до
    # INFO ради отладки голоса. reader пишет строку на каждый RTCP-отчёт
    # (раз в секунду на канал: он считает «неожиданным» всё, кроме
    # ReceiverReport, а Discord штатно шлёт SenderReport), gateway — на
    # каждый кадр голосового WS, потому что discord.py 2.7 добавила туда
    # поле seq, о котором расширение не знает. Обе строки безвредны и обе
    # делают лог нечитаемым.
    for noisy in ("discord.ext.voice_recv.reader", "discord.ext.voice_recv.gateway"):
        logging.getLogger(noisy).setLevel(max(level, logging.WARNING))


@contextmanager
def stage(logger: logging.Logger, name: str, **extra: Any) -> Iterator[dict[str, Any]]:
    """Замерить стадию пайплайна и залогировать её длительность.

    В yield отдаётся словарь: положите туда что угодно, оно попадёт в лог —
    например ``info["chars"] = len(text)``.
    """
    started = time.perf_counter()
    info: dict[str, Any] = dict(extra)
    try:
        yield info
    except Exception:
        elapsed = (time.perf_counter() - started) * 1000
        logger.exception("%s: ОШИБКА через %.0f мс%s", name, elapsed, _fmt(info))
        raise
    else:
        elapsed = (time.perf_counter() - started) * 1000
        logger.info("%s: %.0f мс%s", name, elapsed, _fmt(info))


def _fmt(info: dict[str, Any]) -> str:
    if not info:
        return ""
    return " (" + ", ".join(f"{k}={v}" for k, v in info.items()) + ")"


def guard(name: str, *, reraise: bool = True) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Декоратор для фоновых корутин: падение логируется, а не теряется.

    CancelledError пробрасывается как есть — это штатное завершение.

    ``reraise=False`` — для вечных циклов: их таск никто не ждёт, так что
    пробрасывать исключение некуда, а в логе оно и так уже есть.
    """

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        logger = logging.getLogger(func.__module__)

        @functools.wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            try:
                return await func(*args, **kwargs)
            except asyncio.CancelledError:
                logger.debug("%s: отменён", name)
                raise
            except Exception:
                logger.exception("%s: упал с необработанным исключением", name)
                if reraise:
                    raise
                return None

        return wrapper

    return decorator
