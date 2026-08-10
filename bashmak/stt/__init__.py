"""STT: выбор движка распознавания и общий тип расшифровки.

Движков два, переключаются ключом ``stt.engine``:

* ``gigaam``  — Conformer от SberDevices в onnxruntime, движок по умолчанию;
* ``whisper`` — faster-whisper, прежний движок, оставлен как путь отката.

Второй нужен не ради симметрии. Распознавание — единственная стадия, где
поломка не видна в логе: бот не падает, он просто перестаёт понимать людей.
Откат правкой одной строки конфига, без перекачивания весов, здесь дороже
стоит, чем сэкономленный на удалении файла порядок.

:class:`Transcript` живёт здесь, а не в модуле движка: иначе ``listener.py``
импортировал бы тип расшифровки из ``whisper_worker`` даже в сборке, где
whisper не участвует, и удаление движка ломало бы аудиотракт.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:  # pragma: no cover — только для подсказок типов
    from ..audio.vad import Segment

log = logging.getLogger(__name__)


@dataclass(slots=True)
class Transcript:
    user_id: int
    text: str
    duration: float
    #: Средний логарифм вероятности по токенам. Шкала своя у каждого движка:
    #: у whisper это ``avg_logprob`` сегмента, у GigaAM — среднее по токенам
    #: из ``with_timestamps``. Сравнивать значения между движками нельзя,
    #: порог ``min_avg_logprob`` настраивается под конкретный.
    avg_logprob: float
    #: time.monotonic() на момент конца фразы — переносится из Segment как есть.
    #: Именно по нему очередь LLM расставляет приоритет: «кто первым договорил,
    #: тому первым и отвечаем». Если подставить сюда текущее время, порядок
    #: начнёт определять скорость распознавания — ровно то, чего очередь
    #: пытается избежать. Без значения по умолчанию намеренно: забытое поле
    #: должно падать на месте, а не превращаться в ноль и тихо ломать порядок.
    ended_at: float


class SttPool(Protocol):
    """Что от движка нужно остальному боту — listener.py и doctor.py."""

    async def transcribe(self, segment: Segment) -> Transcript | None: ...

    def close(self) -> None: ...


def create_stt_pool(cfg) -> SttPool:  # noqa: ANN001 — bashmak.config.Section
    """Собрать пул распознавания по ``stt.engine``.

    Настройки движков лежат каждая в своей подсекции, а не вперемешку в общей.
    Так переключение — правда одна строка ``engine:``, а не строка плюс сверка
    полудюжины ключей; к тому же одноимённые настройки у движков означают
    разное (``min_avg_logprob`` считается по своей шкале у каждого, ``workers``
    у whisper это процессы, у GigaAM потоки), и общая секция склеивала бы то,
    что складывать нельзя.

    Импорт движка отложенный, и это не микрооптимизация: whisper тянет
    ctranslate2, GigaAM — onnx-asr, и требовать в окружении оба ради выбора
    одного значило бы вернуть ту самую зависимость, ради ухода от которой
    движок и менялся.
    """
    engine = str(cfg.get("engine", "gigaam")).strip().lower()
    if engine not in ("gigaam", "whisper"):
        raise ValueError(f"неизвестный stt.engine: {engine!r} (ожидается gigaam или whisper)")

    params = cfg.get(engine)
    if params is None or not hasattr(params, "get"):
        raise ValueError(
            f"в конфиге нет секции stt.{engine} с настройками движка "
            "(сверьтесь с config.example.yaml)"
        )

    if engine == "gigaam":
        from .gigaam_worker import GigaamPool

        return GigaamPool(params)

    from .whisper_worker import WhisperPool

    return WhisperPool(params)
