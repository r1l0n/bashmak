"""STT: faster-whisper в пуле процессов.

Распознавание — CPU-bound, и внутри одного процесса Python GIL не даст двум
спикерам считаться параллельно. Поэтому пул именно процессов, а не потоков,
и в каждом процессе своя копия модели (``small`` int8 — около 250 МБ, при
40 ГБ RAM это ничто).

Контекст создаётся через ``spawn``: fork процесса, в котором уже крутится
asyncio-луп и открыты сокеты Discord, — источник трудноуловимых зависаний.
"""

from __future__ import annotations

import asyncio
import logging
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..audio.vad import Segment
from ..utils.logging import stage

log = logging.getLogger(__name__)

# Живёт внутри каждого процесса-воркера, создаётся один раз в инициализаторе.
_MODEL = None


def _init_worker(model_path: str, compute_type: str, cpu_threads: int) -> None:
    global _MODEL
    from faster_whisper import WhisperModel

    _MODEL = WhisperModel(
        model_path,
        device="cpu",
        compute_type=compute_type,
        cpu_threads=cpu_threads,
        num_workers=1,
    )


def _transcribe(
    audio_bytes: bytes,
    language: str,
    beam_size: int,
    no_speech_threshold: float,
    initial_prompt: str | None,
) -> tuple[str, float, float]:
    """Выполняется в процессе-воркере. Возвращает (текст, avg_logprob, no_speech_prob)."""
    if _MODEL is None:  # pragma: no cover — возможно только при поломке пула
        raise RuntimeError("модель STT не инициализирована в воркере")

    audio = np.frombuffer(audio_bytes, dtype=np.float32)
    segments, _info = _MODEL.transcribe(
        audio,
        language=language,
        beam_size=beam_size,
        # VAD уже отработал на входе — второй проход только тратит время.
        vad_filter=False,
        # Без этого Whisper на коротких репликах начинает досочинять
        # продолжение предыдущей фразы.
        condition_on_previous_text=False,
        no_speech_threshold=no_speech_threshold,
        initial_prompt=initial_prompt,
    )

    parts: list[str] = []
    logprobs: list[float] = []
    no_speech: list[float] = []
    for segment in segments:
        parts.append(segment.text)
        logprobs.append(segment.avg_logprob)
        no_speech.append(segment.no_speech_prob)

    text = " ".join(part.strip() for part in parts).strip()
    avg_logprob = sum(logprobs) / len(logprobs) if logprobs else -10.0
    no_speech_prob = max(no_speech) if no_speech else 1.0
    return text, avg_logprob, no_speech_prob


@dataclass(slots=True)
class Transcript:
    user_id: int
    text: str
    duration: float
    avg_logprob: float


class SttPool:
    """Асинхронный фасад над пулом процессов."""

    def __init__(self, cfg) -> None:  # noqa: ANN001 — bashmak.config.Section
        model_path = cfg.path("model_path")
        if not model_path.exists():
            raise FileNotFoundError(
                f"нет модели STT: {model_path}. Запустите ./scripts/setup.sh"
            )

        self.language = cfg.get("language", "ru")
        self.beam_size = int(cfg.get("beam_size", 1))
        self.min_avg_logprob = float(cfg.get("min_avg_logprob", -1.0))
        self.no_speech_threshold = float(cfg.get("no_speech_threshold", 0.6))
        self.initial_prompt = cfg.get("initial_prompt") or None

        workers = int(cfg.get("workers", 2))
        self._pool = ProcessPoolExecutor(
            max_workers=workers,
            mp_context=multiprocessing.get_context("spawn"),
            initializer=_init_worker,
            initargs=(
                str(model_path),
                str(cfg.get("compute_type", "int8")),
                int(cfg.get("cpu_threads_per_worker", 2)),
            ),
        )
        log.info("STT: %s, воркеров %d", Path(model_path).name, workers)

    async def transcribe(self, segment: Segment) -> Transcript | None:
        """Распознать сегмент. None — если это оказался не текст, а шум."""
        loop = asyncio.get_running_loop()
        with stage(log, "stt", user=segment.user_id, sec=f"{segment.duration:.1f}") as info:
            text, avg_logprob, no_speech_prob = await loop.run_in_executor(
                self._pool,
                _transcribe,
                segment.audio.astype(np.float32).tobytes(),
                self.language,
                self.beam_size,
                self.no_speech_threshold,
                self.initial_prompt,
            )
            info["chars"] = len(text)

        if not text:
            log.debug("user=%s: пустая расшифровка, пропускаю", segment.user_id)
            return None

        if no_speech_prob > self.no_speech_threshold or avg_logprob < self.min_avg_logprob:
            log.debug(
                "user=%s: расшифровка отброшена (no_speech=%.2f, logprob=%.2f): %r",
                segment.user_id,
                no_speech_prob,
                avg_logprob,
                text,
            )
            return None

        return Transcript(
            user_id=segment.user_id,
            text=text,
            duration=segment.duration,
            avg_logprob=avg_logprob,
        )

    def close(self) -> None:
        self._pool.shutdown(wait=False, cancel_futures=True)
