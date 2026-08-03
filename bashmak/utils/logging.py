"""Логирование.

По логам должно быть видно две вещи: что бот услышал, отправил и ответил —
и на каком шаге сколько времени ушло. Поэтому построчный лог стадий заменён
на один блок по каждой реплике:

    19:00:26 ┌─ балбес ─ 6.1 с
             │ слышу «иди нахуй.»
             │ в ЛЛМ «балбес: иди нахуй.»
             │ ответ «Сам иди нахуй.»
             └─ stt 0.8 ███  intent 0.0   llm 4.2 ████████████  tts 1.1 ████

Механика прежняя: каждой реплике присваивается correlation id (`cid`) вида
``u123456@41.2``, он контекстной переменной протаскивается через все стадии
(sink → vad → stt → wake → intent → llm → tts → out). Только теперь стадии
не пишут по строке каждая, а копятся в :class:`_Turn` под своим cid и
печатаются разом, когда реплика отработала.

Фоновые задачи оборачиваются в :func:`guard`, чтобы упавший таск не умирал
молча (asyncio по умолчанию проглатывает исключения в задачах).
"""

from __future__ import annotations

import asyncio
import contextvars
import functools
import logging
import logging.handlers
import re
import time
from collections import OrderedDict, deque
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterator

#: Идентификатор обрабатываемой сейчас реплики. Пустой — вне пайплайна.
current_cid: contextvars.ContextVar[str] = contextvars.ContextVar("cid", default="-")

_DATE_FORMAT = "%H:%M:%S"


class _CidFilter(logging.Filter):
    """Подставляет cid в каждую запись, даже если её сделала чужая библиотека."""

    def filter(self, record: logging.LogRecord) -> bool:
        if not hasattr(record, "cid"):
            record.cid = current_cid.get()
        return True


class _Pretty(logging.Formatter):
    """Время и текст — и ничего больше, пока всё идёт хорошо.

    Имя модуля и cid в обычной строке — это тридцать символов шума перед
    каждой репликой. Они нужны, когда что-то сломалось или когда включили
    DEBUG, — там их и показываем.
    """

    def format(self, record: logging.LogRecord) -> str:
        stamp = self.formatTime(record, _DATE_FORMAT)
        if record.levelno >= logging.WARNING:
            head = f"{stamp} {record.levelname:<7} {record.name}: "
        elif record.levelno <= logging.DEBUG:
            head = f"{stamp} [{getattr(record, 'cid', '-')}] {record.name}: "
        else:
            head = f"{stamp} "

        text = head + record.getMessage()
        if record.exc_info:
            text += "\n" + self.formatException(record.exc_info)
        if record.stack_info:
            text += "\n" + self.formatStack(record.stack_info)
        return text


#: Сколько реплик держать недособранными. Реплика живёт секунды, но не всякая
#: доходит до отчёта (мимо бота, ошибка) — лимит страхует от утечки.
_MAX_OPEN_TURNS = 64
#: Сколько последних замеров хранить для сводки в /status.
_RECENT_TURNS = 200

#: Стадии tts приходят по кускам (tts[0], tts[1], ...) — в отчёте это одна.
_STAGE_INDEX = re.compile(r"\[\d+\]$")

_STAGE_TITLES = {"stt": "stt", "intent": "intent", "intent-llm": "intent", "llm": "llm", "tts": "tts"}


@dataclass
class _Turn:
    """Всё, что известно про одну реплику, пока её обрабатывают."""

    started: float
    speaker: str = ""
    heard: str = ""
    sent: str = ""
    reply: str = ""
    stages: "OrderedDict[str, float]" = field(default_factory=OrderedDict)


_open_turns: "OrderedDict[str, _Turn]" = OrderedDict()
_recent: deque[dict[str, float]] = deque(maxlen=_RECENT_TURNS)

log = logging.getLogger("bashmak.turn")


def _turn() -> _Turn:
    cid = current_cid.get()
    turn = _open_turns.get(cid)
    if turn is None:
        turn = _Turn(started=time.perf_counter())
        _open_turns[cid] = turn
        while len(_open_turns) > _MAX_OPEN_TURNS:
            _open_turns.popitem(last=False)
    return turn


def turn_note(**fields: str) -> None:
    """Дописать в отчёт по текущей реплике: speaker, heard, sent, reply."""
    turn = _turn()
    for name, value in fields.items():
        if hasattr(turn, name):
            setattr(turn, name, value)


def turn_drop() -> None:
    """Забыть реплику: она не наша (мимо бота) и отчёта не будет."""
    _open_turns.pop(current_cid.get(), None)


def _clip(text: str, limit: int = 160) -> str:
    text = " ".join(str(text).split())
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _bar(seconds: float, total: float, width: int = 28) -> str:
    if total <= 0 or seconds <= 0:
        return ""
    return "█" * max(1, round(width * seconds / total))


def turn_report() -> None:
    """Напечатать блок по реплике и закрыть её."""
    turn = _open_turns.pop(current_cid.get(), None)
    if turn is None:
        return

    total = time.perf_counter() - turn.started
    lines = [f"┌─ {turn.speaker or 'кто-то'} ─ {total:.1f} с"]
    for label, value in (("слышу", turn.heard), ("в ЛЛМ", turn.sent), ("ответ", turn.reply)):
        if value:
            lines.append(f"│ {label} «{_clip(value)}»")

    measured = sum(turn.stages.values()) or total
    timings = "  ".join(
        f"{name} {sec:.1f} {_bar(sec, measured)}".rstrip() for name, sec in turn.stages.items()
    )
    lines.append(f"└─ {timings}" if timings else "└─")

    log.info("\n".join(lines))
    if turn.stages:
        _recent.append({**turn.stages, "всего": total})


def stage_summary(width: int = 20) -> list[str]:
    """Сводка задержек по последним репликам — для /status."""
    if not _recent:
        return ["замеров пока нет"]

    names: list[str] = []
    for sample in _recent:
        for name in sample:
            if name not in names:
                names.append(name)

    medians: dict[str, float] = {}
    for name in names:
        values = sorted(sample[name] for sample in _recent if name in sample)
        medians[name] = values[len(values) // 2]

    scale = max(medians.values()) or 1.0
    lines = [f"задержки по {len(_recent)} репликам (медиана):"]
    for name in names:
        bar = "█" * max(1, round(width * medians[name] / scale))
        lines.append(f"  {name:<7} {medians[name]:>5.1f} с {bar}")
    return lines


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
    formatter = _Pretty(datefmt=_DATE_FORMAT)
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
    # httpcore отдельно: это транспорт под httpx, и на DEBUG он пишет по
    # десятку строк на каждый запрос к llama-server.
    for noisy in ("httpx", "httpcore"):
        logging.getLogger(noisy).setLevel(max(level, logging.WARNING))

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
        elapsed = time.perf_counter() - started
        _record_stage(name, elapsed)
        logger.exception("%s: ОШИБКА через %.1f с%s", name, elapsed, _fmt(info))
        raise
    else:
        elapsed = time.perf_counter() - started
        _record_stage(name, elapsed)
        # Построчно — только на DEBUG: на INFO стадии уезжают в общий отчёт
        # по реплике, иначе одна фраза занимает пол-экрана.
        logger.debug("%s: %.1f с%s", name, elapsed, _fmt(info))


def _record_stage(name: str, seconds: float) -> None:
    """Приплюсовать время стадии к отчёту по текущей реплике."""
    key = _STAGE_INDEX.sub("", name)
    key = _STAGE_TITLES.get(key, key)
    stages = _turn().stages
    stages[key] = stages.get(key, 0.0) + seconds


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
