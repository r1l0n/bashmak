"""STT: GigaAM (SberDevices) через onnx-asr поверх onnxruntime.

Пул потоков, а не процессов, как у whisper. Процессы там нужны из-за GIL:
faster-whisper держит его почти всё время инференса, и два говорящих внутри
одного процесса считались бы по очереди. onnxruntime GIL на время ``Run``
отпускает, поэтому здесь хватает потоков — и это снимает сразу три платы,
которые пул процессов брал:

* веса лежат в памяти в одном экземпляре, а не по копии на воркера;
* нет спавна интерпретатора, поэтому первая фраза после рестарта не платит
  за подъём пула — модель грузится на старте бота, синхронно и один раз;
* сегмент уходит в модель как есть, без сериализации в байты и пикла через
  границу процесса.

Главное же отличие в самой модели. Энкодер whisper всегда считает окно,
дополненное до 30 секунд, и реплика на две секунды обходится у него почти как
на двадцать. Conformer считает по фактической длине, а в голосовом чате почти
все реплики короткие, — отсюда основной выигрыш, а не из размера модели.

Число потоков здесь означает не то же, что у whisper. Там это были потоки на
воркера, и при двух говорящих машина получала вдвое больше нагрузки, чем при
одном. Здесь ``threads`` — общий пул onnxruntime на все одновременные
распознавания разом, то есть жёсткий потолок, а не множитель.
"""

from __future__ import annotations

import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import numpy as np

from ..audio.resample import WORK_RATE
from ..audio.vad import Segment
from ..utils.logging import stage
from . import Transcript

log = logging.getLogger(__name__)


class GigaamPool:
    """Асинхронный фасад над одной сессией onnxruntime."""

    def __init__(self, cfg) -> None:  # noqa: ANN001 — bashmak.config.Section
        import onnx_asr
        import onnxruntime as rt

        model_path = cfg.path("model_path")
        if not model_path.exists():
            raise FileNotFoundError(
                f"нет модели STT: {model_path}. Запустите ./scripts/setup.sh"
            )

        self.name = str(cfg.get("model", "gigaam-v3-e2e-rnnt"))
        self.min_avg_logprob = float(cfg.get("min_avg_logprob", -1.5))

        # none/пусто в конфиге — это отсутствие квантизации, а не имя варианта.
        quantization = str(cfg.get("quantization", "none")).strip().lower()
        quantization = quantization if quantization not in ("", "none") else None

        threads = int(cfg.get("threads", 3))
        workers = int(cfg.get("workers", 2))

        options = rt.SessionOptions()
        options.intra_op_num_threads = threads
        # Межоператорный пул нужен только в режиме ORT_PARALLEL, которого здесь
        # нет; без явной единицы onnxruntime заводит потоки, которые простаивают.
        options.inter_op_num_threads = 1
        # Без этого потоки onnxruntime между запросами не засыпают, а крутятся
        # в ожидании следующего: замерено — целое ядро горит вхолостую всё
        # время, пока бот сидит в канале. Выигрыш от спина здесь 0.01 с на
        # реплику, а платит за него llama-server своими семью потоками на
        # соседних ядрах. Пул процессов у whisper этой проблемы не имел:
        # процесс между вызовами просто спал.
        options.add_session_config_entry("session.intra_op.allow_spinning", "0")

        try:
            model = onnx_asr.load_model(
                self.name,
                path=str(model_path),
                quantization=quantization,
                sess_options=options,
                providers=["CPUExecutionProvider"],
            )
        except Exception as exc:
            raise RuntimeError(
                f"не удалось поднять модель STT {self.name!r} из {model_path}: {exc}. "
                "Проверьте stt.model и stt.quantization в config.yaml или "
                "перекачайте веса через ./scripts/setup.sh --force"
            ) from exc

        # with_timestamps даёт по токенам logprob — это единственный сигнал
        # уверенности, который отдаёт onnx-asr, и на нём держится отсев шума
        # (см. min_avg_logprob ниже). Обычный recognize возвращает голую строку.
        self._model = model.with_timestamps()
        self._pool = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="stt")

        # Первый Run досоздаёт буферы и прогревает пул потоков onnxruntime.
        # У whisper это доставалось первой живой фразе; здесь платим на старте.
        self._model.recognize(np.zeros(WORK_RATE // 2, dtype=np.float32), sample_rate=WORK_RATE)

        log.info(
            "STT: %s%s, потоков %d, воркеров %d",
            self.name,
            f" ({quantization})" if quantization else "",
            threads,
            workers,
        )

    def _recognize(self, audio: np.ndarray) -> Any:
        return self._model.recognize(audio, sample_rate=WORK_RATE)

    async def transcribe(self, segment: Segment) -> Transcript | None:
        """Распознать сегмент. None — если это оказался не текст, а шум."""
        loop = asyncio.get_running_loop()
        with stage(log, "stt", user=segment.user_id, sec=f"{segment.duration:.1f}") as info:
            result = await loop.run_in_executor(self._pool, self._recognize, segment.audio)
            text = result.text.strip()
            info["chars"] = len(text)

        # На тишине и шуме модель возвращает пустую строку — это и есть её
        # аналог no_speech_prob у whisper, отдельного порога не требуется.
        if not text:
            log.debug("user=%s: пустая расшифровка, пропускаю", segment.user_id)
            return None

        logprobs = getattr(result, "logprobs", None) or []
        avg_logprob = sum(logprobs) / len(logprobs) if logprobs else 0.0

        if avg_logprob < self.min_avg_logprob:
            log.debug(
                "user=%s: расшифровка отброшена (logprob=%.2f < %.2f): %r",
                segment.user_id,
                avg_logprob,
                self.min_avg_logprob,
                text,
            )
            return None

        return Transcript(
            user_id=segment.user_id,
            text=text,
            duration=segment.duration,
            avg_logprob=avg_logprob,
            ended_at=segment.ended_at,
        )

    def close(self) -> None:
        self._pool.shutdown(wait=False, cancel_futures=True)
