"""Кастомный AudioSink py-cord: разводка входящего голоса по говорящим.

py-cord отдаёт голос уже раздельно по SSRC — смешивать ничего не нужно,
достаточно разложить кадры по буферам и не тормозить.

``write()`` вызывается из потока приёма голоса, общего на весь голосовой
клиент. Всё, что здесь делается, — микс стерео в моно, ресемпл 48→16 кГц и
append в дек. VAD, STT и прочее живут в asyncio-лупе (см. listener.py).
"""

from __future__ import annotations

import logging

from discord.sinks import Sink

from .buffer import StreamRegistry
from .resample import DISCORD_RATE, WORK_RATE, StreamResampler, discord_pcm_to_mono

log = logging.getLogger(__name__)


class BashmakSink(Sink):
    """Sink, который ничего не пишет на диск — только раздаёт PCM пайплайну."""

    def __init__(self, registry: StreamRegistry) -> None:
        super().__init__()
        self.registry = registry
        self._resamplers: dict[int, StreamResampler] = {}
        self._warned: set[int] = set()

    # py-cord зовёт это на каждые 20 мс речи каждого участника.
    def write(self, data, user) -> None:  # noqa: ANN001 — сигнатура задана py-cord
        if user is None or not data:
            return

        user_id = getattr(user, "id", user)
        try:
            mono48 = discord_pcm_to_mono(data)
            if mono48.size == 0:
                return

            resampler = self._resamplers.get(user_id)
            if resampler is None:
                resampler = StreamResampler(DISCORD_RATE, WORK_RATE)
                self._resamplers[user_id] = resampler

            mono16 = resampler.process(mono48)
            if mono16.size:
                self.registry.get(user_id).push(mono16)
        except Exception:
            # Исключение здесь убило бы поток приёма для ВСЕХ участников,
            # поэтому логируем один раз на пользователя и продолжаем.
            if user_id not in self._warned:
                self._warned.add(user_id)
                log.exception("сбой обработки входящего кадра, user=%s", user_id)

    def forget(self, user_id: int) -> None:
        """Забыть состояние ушедшего участника."""
        self._resamplers.pop(user_id, None)
        self._warned.discard(user_id)
        self.registry.drop(user_id)

    # Базовый Sink на этом этапе склеивает wav-файлы — нам это не нужно.
    def cleanup(self) -> None:
        self.finished = True
        self._resamplers.clear()
        self.registry.clear()

    def format_audio(self, audio) -> None:  # noqa: ANN001 — сигнатура задана py-cord
        return None
