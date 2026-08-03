"""Потребитель аудио: буферы → VAD → STT.

Один asyncio-таск опрашивает буферы всех говорящих. Опрос, а не будильник из
потока приёма, — сознательно: при десятке участников это доли процента CPU,
зато нет ни одного примитива синхронизации между потоком роутера пакетов и
лупом, то есть нет и класса багов «залип на локе внутри voice receive».

Задержка опроса (30 мс) на порядок меньше порога тишины (700 мс), так что на
момент нарезки фраз она не влияет.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Awaitable, Callable

from ..stt.whisper_worker import SttPool, Transcript
from ..utils.logging import current_cid, guard, new_cid
from .buffer import StreamRegistry
from .sink import BashmakSink
from .vad import Segment, Segmenter, SileroVad

log = logging.getLogger(__name__)

POLL_INTERVAL = 0.03
#: Столько молчит участник — закрываем его сегментер и освобождаем память.
IDLE_DROP_SECONDS = 60.0
#: Максимум окон VAD на одного говорящего за тик опроса.
#:
#: Инференс Silero синхронный и считается прямо в лупе. В штатном режиме это
#: одно окно (32 мс речи) на тик (30 мс) — доли процента CPU. Но буфер держит
#: до 30 секунд, и после любой заминки (GC, спавн воркера STT) один вызов
#: прогнал бы под тысячу инференсов подряд и заблокировал луп на сотни
#: миллисекунд — а это пропущенные heartbeat'ы Discord и разрыв соединения.
#: С лимитом отставание разгребается за несколько тиков, по 16 окон за раз.
MAX_VAD_WINDOWS_PER_TICK = 16


class VoiceListener:
    """Владеет sink'ом, VAD-инстансами и раздаёт расшифровки наружу."""

    def __init__(
        self,
        cfg,  # noqa: ANN001 — bashmak.config.Config
        stt: SttPool,
        on_transcript: Callable[[Transcript], Awaitable[None]],
    ) -> None:
        self._cfg = cfg
        self._vad_cfg = cfg.vad
        self._stt = stt
        self._on_transcript = on_transcript

        self.registry = StreamRegistry(rate=int(cfg.audio.work_sample_rate))
        self.sink = BashmakSink(self.registry)

        self._segmenters: dict[int, Segmenter] = {}
        self._task: asyncio.Task | None = None
        self._jobs: set[asyncio.Task] = set()

        # Модель грузится один раз, состояние у каждого говорящего своё.
        self._vad_path = self._vad_cfg.path("model_path")
        self._window = int(self._vad_cfg.get("window_samples", 512))
        # Discord перестаёт слать пакеты, когда человек замолчал, — тишины,
        # по которой VAD закрыл бы фразу, физически не приходит. Поэтому
        # закрываем её сами, продержав паузу чуть дольше порога VAD.
        self._hangover = float(self._vad_cfg.get("silence_ms", 700)) / 1000 + 0.2

    def start(self) -> None:
        if self._task is None or self._task.done():
            self._task = asyncio.create_task(self._run(), name="voice-listener")
            log.info("слушатель голоса запущен")

    async def stop(self) -> None:
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None

        jobs = list(self._jobs)
        for job in jobs:
            job.cancel()
        if jobs:
            # Именно дождаться, а не только отменить: иначе на остановке
            # прилетит «Task was destroyed but it is pending».
            await asyncio.gather(*jobs, return_exceptions=True)
        self._jobs.clear()
        self._segmenters.clear()
        self.registry.clear()
        log.info("слушатель голоса остановлен")

    def _segmenter_for(self, user_id: int) -> Segmenter:
        segmenter = self._segmenters.get(user_id)
        if segmenter is None:
            segmenter = Segmenter(
                user_id,
                SileroVad(self._vad_path, int(self._cfg.audio.work_sample_rate)),
                threshold=float(self._vad_cfg.get("threshold", 0.5)),
                silence_ms=int(self._vad_cfg.get("silence_ms", 700)),
                min_speech_ms=int(self._vad_cfg.get("min_speech_ms", 300)),
                max_segment_s=float(self._vad_cfg.get("max_segment_s", 20)),
                preroll_ms=int(self._vad_cfg.get("preroll_ms", 300)),
                window_samples=self._window,
            )
            self._segmenters[user_id] = segmenter
        return segmenter

    @guard("слушатель голоса", reraise=False)
    async def _run(self) -> None:
        while True:
            await asyncio.sleep(POLL_INTERVAL)
            try:
                streams = self.registry.snapshot()
            except Exception:
                log.exception("не смог получить список источников аудио")
                continue
            for stream in streams:
                try:
                    self._process_stream(stream)
                except Exception:
                    # Один сломавшийся спикер не должен глушить остальных.
                    log.exception("user=%s: сбой в пайплайне VAD", stream.user_id)

    def _process_stream(self, stream) -> None:  # noqa: ANN001 — buffer.UserStream
        samples = stream.drain()
        segmenter = self._segmenter_for(stream.user_id)

        # pending — остаток, не разобранный на прошлом тике из-за лимита окон.
        segments = (
            segmenter.feed(samples, max_windows=MAX_VAD_WINDOWS_PER_TICK)
            if samples.size or segmenter.pending
            else []
        )

        if not samples.size and not segmenter.pending:
            idle = stream.idle_seconds
            if segmenter.in_speech and idle > self._hangover:
                tail = segmenter.flush()
                if tail is not None:
                    log.debug("user=%s: фраза закрыта по молчанию канала", stream.user_id)
                    segments.append(tail)
            elif idle > IDLE_DROP_SECONDS:
                self._segmenters.pop(stream.user_id, None)
                self.sink.forget(stream.user_id)

        for segment in segments:
            self._dispatch(segment)

    def _dispatch(self, segment: Segment) -> None:
        """Отправить фразу в STT, не блокируя опрос буферов."""
        cid = new_cid(segment.user_id)
        job = asyncio.create_task(self._handle(segment, cid), name=f"stt-{cid}")
        self._jobs.add(job)
        job.add_done_callback(self._jobs.discard)

    async def _handle(self, segment: Segment, cid: str) -> None:
        current_cid.set(cid)
        log.debug("фраза закрыта: user=%s, %.1f с", segment.user_id, segment.duration)
        try:
            transcript = await self._stt.transcribe(segment)
        except Exception:
            log.exception("user=%s: STT не смог обработать сегмент", segment.user_id)
            return

        if transcript is None:
            return

        log.info("распознано: %r", transcript.text)
        try:
            await self._on_transcript(transcript)
        except Exception:
            log.exception("обработчик расшифровки упал")
