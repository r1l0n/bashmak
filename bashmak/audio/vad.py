"""Silero VAD и нарезка потока на фразы.

Модель гоняется напрямую в onnxruntime, минуя pip-пакет ``silero-vad``: тот
тянет за собой torch + torchaudio (~800 МБ) ради двух мегабайт весов. Сам
``.onnx`` качает scripts/setup.sh.

Инстанс VAD — свой на каждого говорящего: модель рекуррентная, и общее
состояние на всех превратило бы речь двух человек в кашу.
"""

from __future__ import annotations

import logging
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)

#: Silero v5 принимает строго 512 сэмплов на окно при 16 кГц (32 мс).
WINDOW_SAMPLES_16K = 512


@dataclass(slots=True)
class Segment:
    """Готовая фраза одного говорящего."""

    user_id: int
    audio: np.ndarray  # float32 моно 16 кГц
    duration: float  # секунды
    ended_at: float  # time.monotonic() на момент конца фразы


class SileroVad:
    """Тонкая обёртка над silero_vad.onnx. Держит рекуррентное состояние."""

    def __init__(self, model_path: str | Path, sample_rate: int = 16000) -> None:
        # Импорт внутри конструктора: Segmenter — чистая логика, и модуль должен
        # импортироваться (и тестироваться) без установленного рантайма.
        import onnxruntime as ort

        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(
                f"нет модели VAD: {model_path}. Запустите ./scripts/setup.sh"
            )

        options = ort.SessionOptions()
        # VAD считается на каждые 32 мс речи каждого участника — он обязан быть
        # дешёвым. Один тред: пусть ядра достаются STT и LLM.
        options.intra_op_num_threads = 1
        options.inter_op_num_threads = 1
        options.log_severity_level = 3

        self.session = ort.InferenceSession(
            str(model_path), options, providers=["CPUExecutionProvider"]
        )
        self.sample_rate = sample_rate
        self._sr = np.array(sample_rate, dtype=np.int64)

        names = {i.name for i in self.session.get_inputs()}
        # v5 держит одно объединённое состояние, v4 — раздельные h/c.
        self._v5 = "state" in names
        if not self._v5 and not {"h", "c"} <= names:
            raise RuntimeError(f"незнакомая сигнатура модели VAD: {sorted(names)}")
        self.reset()

    def reset(self) -> None:
        """Сбросить состояние — обязательно между фразами."""
        if self._v5:
            self._state = np.zeros((2, 1, 128), dtype=np.float32)
        else:
            self._h = np.zeros((2, 1, 64), dtype=np.float32)
            self._c = np.zeros((2, 1, 64), dtype=np.float32)

    def speech_probability(self, window: np.ndarray) -> float:
        x = np.ascontiguousarray(window, dtype=np.float32).reshape(1, -1)
        if self._v5:
            out, self._state = self.session.run(
                None, {"input": x, "state": self._state, "sr": self._sr}
            )
        else:
            out, self._h, self._c = self.session.run(
                None, {"input": x, "h": self._h, "c": self._c, "sr": self._sr}
            )
        return float(out[0][0])


class Segmenter:
    """Конечный автомат «тишина ↔ речь» поверх VAD для одного говорящего.

    Состояний два. В тишине окна копятся в кольцевой пре-ролл, чтобы при
    старте речи не потерять первый слог (VAD срабатывает уже после начала
    звука). В речи окна накапливаются, пока не наберётся ``silence_ms``
    тишины подряд или сегмент не упрётся в ``max_segment_s``.
    """

    def __init__(
        self,
        user_id: int,
        vad: SileroVad,
        *,
        threshold: float = 0.5,
        silence_ms: int = 700,
        min_speech_ms: int = 300,
        max_segment_s: float = 20.0,
        preroll_ms: int = 300,
        window_samples: int = WINDOW_SAMPLES_16K,
    ) -> None:
        self.user_id = user_id
        self.vad = vad
        self.threshold = threshold
        self.window_samples = window_samples
        self.window_ms = window_samples * 1000 / vad.sample_rate

        self.silence_windows = max(1, round(silence_ms / self.window_ms))
        self.min_speech_windows = max(1, round(min_speech_ms / self.window_ms))
        self.max_windows = max(1, round(max_segment_s * 1000 / self.window_ms))

        self._tail = np.zeros(0, dtype=np.float32)
        # +1, потому что окно, на котором сработал VAD, само лежит в пре-ролле:
        # без запаса «предыдущих» окон осталось бы на одно меньше заказанного.
        self._preroll: deque[np.ndarray] = deque(
            maxlen=max(1, round(preroll_ms / self.window_ms) + 1)
        )
        self._speech: list[np.ndarray] = []
        self._silence_run = 0
        self._speech_windows = 0
        self._in_speech = False

    @property
    def in_speech(self) -> bool:
        return self._in_speech

    def feed(self, samples: np.ndarray) -> list[Segment]:
        """Скормить кусок аудио, получить завершившиеся за это время фразы."""
        segments: list[Segment] = []
        if samples.size:
            self._tail = np.concatenate((self._tail, samples)) if self._tail.size else samples

        while self._tail.size >= self.window_samples:
            window = self._tail[: self.window_samples]
            self._tail = self._tail[self.window_samples :]

            try:
                probability = self.vad.speech_probability(window)
            except Exception:
                # Падение VAD не должно валить приём аудио: считаем окно тишиной
                # и идём дальше — хуже быть не может, а поток жив.
                log.exception("user=%s: VAD упал на окне, считаю его тишиной", self.user_id)
                probability = 0.0

            if not self._in_speech:
                self._preroll.append(window)
                if probability >= self.threshold:
                    self._in_speech = True
                    self._speech = list(self._preroll)
                    self._preroll.clear()
                    self._silence_run = 0
                    self._speech_windows = 1
                continue

            self._speech.append(window)
            if probability >= self.threshold:
                self._silence_run = 0
                self._speech_windows += 1
            else:
                self._silence_run += 1

            if self._silence_run >= self.silence_windows:
                segment = self._close(trim_silence=True)
                if segment is not None:
                    segments.append(segment)
            elif len(self._speech) >= self.max_windows:
                # Длинный монолог режем принудительно, чтобы STT не ждал вечно.
                log.debug("user=%s: сегмент упёрся в max_segment_s", self.user_id)
                segment = self._close(trim_silence=False)
                if segment is not None:
                    segments.append(segment)

        return segments

    def flush(self) -> Segment | None:
        """Закрыть незавершённую фразу — например, когда человек вышел из канала."""
        if not self._in_speech:
            return None
        return self._close(trim_silence=True)

    def _close(self, *, trim_silence: bool) -> Segment | None:
        windows = self._speech
        speech_windows = self._speech_windows
        silence_run = self._silence_run

        self._speech = []
        self._in_speech = False
        self._silence_run = 0
        self._speech_windows = 0
        self._preroll.clear()
        # Состояние рекуррентное: без сброса «хвост» прошлой фразы влияет на следующую.
        self.vad.reset()

        if not windows:
            return None

        if speech_windows < self.min_speech_windows:
            log.debug(
                "user=%s: сегмент %d окон — короче min_speech_ms, выброшен",
                self.user_id,
                speech_windows,
            )
            return None

        if trim_silence and silence_run > 1:
            # Хвост тишины Whisper не нужен, а время на него потратит.
            windows = windows[: len(windows) - (silence_run - 1)]

        audio = np.concatenate(windows)
        return Segment(
            user_id=self.user_id,
            audio=audio,
            duration=audio.size / self.vad.sample_rate,
            ended_at=time.monotonic(),
        )
