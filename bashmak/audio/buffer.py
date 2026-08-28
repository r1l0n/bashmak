"""Пер-юзерные буферы аудио.

Единственное место, где встречаются поток роутера пакетов
discord-ext-voice-recv (пишет) и asyncio-луп бота (читает). Писатель только
кладёт готовый numpy-массив под коротким локом: поток приёма общий на весь
голосовой клиент, и задержка в нём тормозит речь всех участников сразу.
"""

from __future__ import annotations

import logging
import threading
import time
from collections import deque

import numpy as np

log = logging.getLogger(__name__)


class UserStream:
    """Очередь аудио-чанков одного говорящего."""

    def __init__(self, user_id: int, max_seconds: float = 30.0, rate: int = 16000) -> None:
        self.user_id = user_id
        self.last_write = time.monotonic()

        self._lock = threading.Lock()
        self._chunks: deque[np.ndarray] = deque()
        self._samples = 0
        self._rate = max(1, int(rate))
        self._max_samples = int(max_seconds * rate)
        self._dropped = 0

    def push(self, samples: np.ndarray) -> None:
        """Вызывается из потока приёма голоса. Обязан быть быстрым."""
        if samples.size == 0:
            return
        with self._lock:
            self._chunks.append(samples)
            self._samples += samples.size
            # Если потребитель отстал (например, залип STT), лучше потерять
            # самое старое аудио, чем съесть всю память.
            while self._samples > self._max_samples:
                old = self._chunks.popleft()
                self._samples -= old.size
                self._dropped += old.size
        self.last_write = time.monotonic()

    def drain(self) -> np.ndarray:
        """Забрать всё накопленное. Вызывается из asyncio-лупа."""
        with self._lock:
            if not self._chunks:
                return np.zeros(0, dtype=np.float32)
            chunks = list(self._chunks)
            dropped = self._dropped
            self._chunks.clear()
            self._samples = 0
            self._dropped = 0

        if dropped:
            log.warning(
                "user=%s: потребитель не успевает, выброшено %.1f с аудио",
                self.user_id,
                dropped / self._rate,
            )
        return np.concatenate(chunks)

    @property
    def idle_seconds(self) -> float:
        return time.monotonic() - self.last_write


class StreamRegistry:
    """user_id → UserStream, с созданием по требованию и уборкой молчунов."""

    def __init__(self, max_seconds: float = 30.0, rate: int = 16000) -> None:
        self._lock = threading.Lock()
        self._streams: dict[int, UserStream] = {}
        self._max_seconds = max_seconds
        self._rate = rate

    def get(self, user_id: int) -> UserStream:
        with self._lock:
            stream = self._streams.get(user_id)
            if stream is None:
                stream = UserStream(user_id, self._max_seconds, self._rate)
                self._streams[user_id] = stream
                log.debug("новый источник аудио: user=%s", user_id)
            return stream

    def snapshot(self) -> list[UserStream]:
        with self._lock:
            return list(self._streams.values())

    def drop(self, user_id: int) -> None:
        with self._lock:
            if self._streams.pop(user_id, None) is not None:
                log.debug("источник аудио закрыт: user=%s", user_id)

    def clear(self) -> None:
        with self._lock:
            self._streams.clear()
