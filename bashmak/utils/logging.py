"""Логирование.

По логам должно быть видно две вещи: что бот услышал, отправил и ответил, и
сколько времени ушло на каждом шаге. Поэтому построчный лог стадий заменён на
один блок по каждой реплике:

    19:00:26 ┌─ участник ─ 6.8 с
             │ слышу «привет.»
             │ в ЛЛМ «привет.»
             │ ответ «И тебе привет.»
             └─ ожидание 0.7 ██  stt 0.8 ███  intent 0.0   llm 4.2 ████  tts 1.1 ██

Каждой реплике присваивается correlation id (`cid`) вида ``u123456@41.2``, он
контекстной переменной протаскивается через все стадии (sink → vad → stt →
wake → intent → llm → tts → out). Стадии копятся в :class:`_Turn` под своим cid
и печатаются разом, когда реплика отработала.

Отсчёт ведётся от момента, когда человек договорил (:func:`turn_start`), а не
от первой стадии: пауза, по которой фраза закрывается, и ожидание в очереди к
модели — это то время, которое слышно как молчание бота.

Фоновые задачи оборачиваются в :func:`guard`, чтобы упавший таск не умирал
молча: asyncio по умолчанию проглатывает исключения в задачах.
"""

from __future__ import annotations

import asyncio
import contextvars
import functools
import json
import logging
import logging.handlers
import os
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
    """В штатном режиме — только время и текст. Имя модуля и cid добавляют
    тридцать символов перед строкой и показываются лишь при ошибках и DEBUG."""

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

#: intent-llm — тот же шаг разбора намерения, только через модель; в отчёте
#: это одна строка «intent», а не две.
_STAGE_TITLES = {"intent-llm": "intent"}

#: Сколько прошло между «человек договорил» и стартом распознавания: пауза, по
#: которой закрывается фраза (vad.silence_ms плюс запас в listener.py), плюс
#: ожидание свободного воркера STT. Считается снаружи, по Segment.ended_at.
WAIT_STAGE = "ожидание"
#: Сколько реплика пролежала в очереди к LLM за чужим инференсом.
QUEUE_STAGE = "очередь"


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


#: Метрики реплик для монитора (bashmak/monitor.py). Отдельным файлом и в JSON,
#: а не разбором основного лога: монитор — другой процесс, и разбирать обратно
#: уже отформатированный текст ненадёжно.
METRICS_NAME = "turns.jsonl"
#: Живые уровни говорящих — снимок, а не история: монитору важно только
#: «сейчас». Поэтому обычный json, который перезаписывается целиком.
LEVELS_NAME = "levels.json"

_metrics = logging.getLogger("bashmak.metrics")


def metrics_path(cfg: Any = None) -> Path:
    """Где лежит файл метрик. Рядом с основным логом."""
    default = Path(__file__).resolve().parent.parent.parent / "logs" / METRICS_NAME
    if cfg is None:
        return default
    section = cfg.get("logging") if hasattr(cfg, "get") else None
    if section is None or "file" not in section:
        return default
    return section.path("file").with_name(METRICS_NAME)


def levels_path(cfg: Any = None) -> Path:
    """Где лежит снимок уровней. Рядом с метриками."""
    return metrics_path(cfg).with_name(LEVELS_NAME)


def publish_levels(path: Path, payload: dict[str, Any]) -> None:
    """Переписать снимок уровней целиком.

    Через временный файл и os.replace: монитор читает его постоянно и не должен
    получить наполовину записанным. Ошибки подавляются — индикатор не повод
    ронять бота.
    """
    temporary = path.with_suffix(".tmp")
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        os.replace(temporary, path)
    except OSError:
        pass


def _setup_metrics(log_file: Path | None, max_bytes: int, backups: int) -> None:
    """Отдельный логгер под метрики: одна JSON-строка на реплику."""
    _metrics.propagate = False
    _metrics.setLevel(logging.INFO)
    for existing in list(_metrics.handlers):
        _metrics.removeHandler(existing)
    if log_file is None:
        return
    try:
        path = log_file.with_name(METRICS_NAME)
        path.parent.mkdir(parents=True, exist_ok=True)
        handler = logging.handlers.RotatingFileHandler(
            path, maxBytes=max_bytes, backupCount=min(backups, 2), encoding="utf-8"
        )
        handler.setFormatter(logging.Formatter("%(message)s"))
        _metrics.addHandler(handler)
    except OSError as exc:
        logging.getLogger().warning("не удалось открыть файл метрик: %s", exc)


def _turn() -> _Turn:
    cid = current_cid.get()
    turn = _open_turns.get(cid)
    if turn is None:
        turn = _Turn(started=time.perf_counter())
        _open_turns[cid] = turn
        while len(_open_turns) > _MAX_OPEN_TURNS:
            _open_turns.popitem(last=False)
    return turn


def turn_start(waited: float = 0.0) -> None:
    """Открыть реплику, отсчитав её начало от конца фразы.

    Иначе ``_Turn`` заводится на старте STT, и в «всего» не попадает ни пауза,
    по которой фраза закрылась, ни ожидание свободного воркера.

    ``waited`` — разность двух ``time.monotonic()``, а вычитается она из
    ``time.perf_counter()``: используется длительность, а ход у часов одинаковый.
    """
    waited = max(0.0, waited)
    cid = current_cid.get()
    turn = _Turn(started=time.perf_counter() - waited)
    if waited:
        turn.stages[WAIT_STAGE] = waited
    _open_turns[cid] = turn
    while len(_open_turns) > _MAX_OPEN_TURNS:
        _open_turns.popitem(last=False)


def turn_stage(name: str, seconds: float) -> None:
    """Приплюсовать время стадии к отчёту по текущей реплике.

    Зовётся из :func:`stage` и напрямую — для отрезков, внутри которых нашего
    кода нет и реплика просто ждёт (пауза после фразы, очередь к модели).
    """
    key = _STAGE_INDEX.sub("", name)
    key = _STAGE_TITLES.get(key, key)
    stages = _turn().stages
    stages[key] = stages.get(key, 0.0) + seconds


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
        _metrics.info(
            json.dumps(
                {
                    "at": time.time(),
                    "speaker": turn.speaker,
                    "heard": turn.heard,
                    "sent": turn.sent,
                    "reply": turn.reply,
                    "stages": dict(turn.stages),
                    "total": round(total, 3),
                },
                ensure_ascii=False,
            )
        )


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
        lines.append(f"  {name:<9} {medians[name]:>5.1f} с {bar}")
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

    _setup_metrics(log_file, max_bytes, backups)

    # На DEBUG эти логгеры заглушают полезные строки объёмом: httpcore — это
    # транспорт под httpx, он пишет по десятку строк на запрос к llama-server.
    for noisy in ("discord", "httpx", "httpcore"):
        logging.getLogger(noisy).setLevel(max(level, logging.WARNING))

    # Эти двое шумят уже на INFO, то есть даже когда discord поднимают ради
    # отладки голоса. reader пишет строку на каждый RTCP-отчёт (он считает
    # «неожиданным» всё, кроме ReceiverReport, а Discord штатно шлёт
    # SenderReport), gateway — на каждый кадр голосового WS из-за поля seq,
    # добавленного в discord.py 2.7. Обе строки безвредны.
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
        turn_stage(name, elapsed)
        logger.exception("%s: ОШИБКА через %.1f с%s", name, elapsed, _fmt(info))
        raise
    else:
        elapsed = time.perf_counter() - started
        turn_stage(name, elapsed)
        # Построчно только на DEBUG: на INFO стадии уходят в общий отчёт.
        logger.debug("%s: %.1f с%s", name, elapsed, _fmt(info))


def _fmt(info: dict[str, Any]) -> str:
    if not info:
        return ""
    return " (" + ", ".join(f"{k}={v}" for k, v in info.items()) + ")"


def guard(name: str, *, reraise: bool = True) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Декоратор для фоновых корутин: падение логируется, а не теряется.

    CancelledError пробрасывается как есть — это штатное завершение.
    ``reraise=False`` — для вечных циклов: их таск никто не ждёт.
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
