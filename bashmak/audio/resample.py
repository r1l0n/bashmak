"""Пересчёт частоты дискретизации между форматом Discord и форматом моделей.

Discord отдаёт и принимает 48 кГц / 2 канала / s16le. VAD и Whisper работают
с 16 кГц моно float32, Piper синтезирует 22.05 кГц моно int16.

Ресемплинг потоковый (:class:`StreamResampler`), а не покадровый: если гонять
stateless-функцию на каждые 20 мс, на стыках кадров появляются щелчки, и VAD
начинает принимать их за речь.
"""

from __future__ import annotations

import numpy as np
import soxr

DISCORD_RATE = 48000
DISCORD_CHANNELS = 2
WORK_RATE = 16000


class StreamResampler:
    """Ресемплер с сохранением состояния между кадрами. Один на источник."""

    def __init__(self, in_rate: int, out_rate: int) -> None:
        self.in_rate = in_rate
        self.out_rate = out_rate
        self._stream = None
        if in_rate != out_rate:
            # ResampleStream появился не во всех сборках soxr — если его нет,
            # откатываемся на stateless-вариант (чуть хуже на стыках, но работает).
            factory = getattr(soxr, "ResampleStream", None)
            if factory is not None:
                self._stream = factory(in_rate, out_rate, 1, dtype="float32")

    def process(self, mono: np.ndarray) -> np.ndarray:
        if self.in_rate == self.out_rate:
            return mono
        if self._stream is not None:
            return self._stream.resample_chunk(mono)
        return soxr.resample(mono, self.in_rate, self.out_rate)


def discord_pcm_to_mono(data: bytes) -> np.ndarray:
    """48 кГц stereo s16le (как отдаёт py-cord) → 48 кГц моно float32 в [-1, 1]."""
    pcm = np.frombuffer(data, dtype="<i2")
    # Обрезанный кадр (нечётное число сэмплов) не должен ронять поток.
    usable = (pcm.size // DISCORD_CHANNELS) * DISCORD_CHANNELS
    if usable == 0:
        return np.zeros(0, dtype=np.float32)
    stereo = pcm[:usable].reshape(-1, DISCORD_CHANNELS)
    return stereo.mean(axis=1, dtype=np.float32) / 32768.0


def mono_to_discord_pcm(mono: np.ndarray, src_rate: int) -> bytes:
    """Моно (int16 или float32) с частотой src_rate → 48 кГц stereo s16le."""
    if mono.size == 0:
        return b""

    samples = mono.astype(np.float32)
    if mono.dtype == np.int16:
        samples /= 32768.0

    if src_rate != DISCORD_RATE:
        samples = soxr.resample(samples, src_rate, DISCORD_RATE)

    clipped = np.clip(samples * 32767.0, -32768.0, 32767.0).astype("<i2")
    stereo = np.repeat(clipped[:, np.newaxis], DISCORD_CHANNELS, axis=1)
    return stereo.tobytes()
