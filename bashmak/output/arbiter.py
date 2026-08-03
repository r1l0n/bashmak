"""Единственный голосовой выход канала.

В канал физически уходит один поток, а претендентов на него два: ответы бота
и музыка. ТЗ предлагает ставить музыку на паузу, но в py-cord это выходит
криво: ``voice_client`` держит ровно один источник, вызвать ``play()`` поверх
паузы нельзя, а значит пришлось бы останавливать трек и пересоздавать
``FFmpegPCMAudio`` с ``-ss <позиция>`` — то есть рывок, потеря позиции при
живом стриме и лишний коннект к YouTube на каждую реплику.

Поэтому источник ровно один и на всё время: :class:`MixerSource`. Он сам
складывает музыку с речью и приглушает музыку, пока бот говорит. Ничего не
перезапускается, приоритет TTS соблюдён, позиция трека не теряется.

``read()`` вызывается плеером py-cord из отдельного потока каждые 20 мс —
всё, что тут происходит, обязано быть арифметикой над 960 сэмплами.
"""

from __future__ import annotations

import asyncio
import logging
import threading
from typing import AsyncIterator, Callable

import discord
import numpy as np

from ..audio.resample import mono_to_discord_pcm

log = logging.getLogger(__name__)

#: 20 мс при 48 кГц стерео int16 — размер кадра, который ждёт py-cord.
FRAME_SAMPLES = 960
FRAME_BYTES = FRAME_SAMPLES * 2 * 2
SILENCE = bytes(FRAME_BYTES)


class MixerSource(discord.AudioSource):
    """Микс «музыка + речь» с даккингом. Живёт, пока бот в канале."""

    def __init__(
        self,
        *,
        loop: asyncio.AbstractEventLoop,
        volume: float = 0.5,
        duck_volume: float = 0.25,
        fade_ms: int = 150,
        on_music_end: Callable[[], None] | None = None,
    ) -> None:
        self._loop = loop
        self._on_music_end = on_music_end

        self.volume = float(volume)
        self.duck_volume = float(duck_volume)
        # За сколько кадров доехать от полной громкости до приглушённой.
        frames = max(1, round(fade_ms / 20))
        self._fade_step = 1.0 / frames
        self._gain_ratio = 1.0  # 1.0 — обычная громкость, 0.0 — полностью приглушено

        self._music: discord.AudioSource | None = None
        self._music_paused = False
        self._music_lock = threading.Lock()

        self._tts = bytearray()
        self._tts_lock = threading.Lock()

        self.drained = asyncio.Event()
        self.drained.set()

    # ------------------------------------------------------------ речь ----
    def push_tts(self, pcm: bytes) -> None:
        """Добавить синтезированный кусок. Вызывается из лупа."""
        if not pcm:
            return
        with self._tts_lock:
            self._tts.extend(pcm)
        self.drained.clear()

    def clear_tts(self) -> None:
        with self._tts_lock:
            self._tts.clear()
        self._loop.call_soon_threadsafe(self.drained.set)

    @property
    def speaking(self) -> bool:
        with self._tts_lock:
            return bool(self._tts)

    # ---------------------------------------------------------- музыка ----
    def set_music(self, source: discord.AudioSource | None) -> None:
        with self._music_lock:
            old, self._music = self._music, source
            self._music_paused = False
        if old is not None:
            try:
                old.cleanup()
            except Exception:
                log.exception("не удалось закрыть предыдущий музыкальный источник")

    def pause_music(self) -> None:
        with self._music_lock:
            self._music_paused = True

    def resume_music(self) -> None:
        with self._music_lock:
            self._music_paused = False

    @property
    def music_paused(self) -> bool:
        return self._music_paused

    @property
    def has_music(self) -> bool:
        return self._music is not None

    def set_volume(self, value: float) -> float:
        self.volume = max(0.0, min(1.5, float(value)))
        return self.volume

    # --------------------------------------------------------- пайплайн ---
    def is_opus(self) -> bool:
        return False

    def read(self) -> bytes:
        """20 мс микса. Никогда не возвращает b'' — это завершило бы плеер навсегда."""
        try:
            return self._mix()
        except Exception:
            # Молчание на один кадр лучше, чем умерший голосовой выход.
            log.exception("сбой в микшере, отдаю тишину")
            return SILENCE

    def _mix(self) -> bytes:
        speech = self._take_tts()
        music = self._take_music()

        # Даккинг: пока в буфере есть речь, плавно уводим музыку вниз.
        target = 0.0 if speech is not None else 1.0
        if self._gain_ratio < target:
            self._gain_ratio = min(target, self._gain_ratio + self._fade_step)
        elif self._gain_ratio > target:
            self._gain_ratio = max(target, self._gain_ratio - self._fade_step)

        music_gain = self.duck_volume + (self.volume - self.duck_volume) * self._gain_ratio

        if speech is None and music is None:
            return SILENCE
        if music is None:
            return speech  # речь идёт как есть, на полной громкости
        if speech is None and abs(music_gain - 1.0) < 1e-3:
            return music

        mixed = np.zeros(FRAME_SAMPLES * 2, dtype=np.float32)
        if music is not None:
            mixed += np.frombuffer(music, dtype="<i2").astype(np.float32) * music_gain
        if speech is not None:
            mixed += np.frombuffer(speech, dtype="<i2").astype(np.float32)

        np.clip(mixed, -32768.0, 32767.0, out=mixed)
        return mixed.astype("<i2").tobytes()

    def _take_tts(self) -> bytes | None:
        with self._tts_lock:
            if not self._tts:
                return None
            if len(self._tts) >= FRAME_BYTES:
                frame = bytes(self._tts[:FRAME_BYTES])
                del self._tts[:FRAME_BYTES]
            else:
                # Хвост короче кадра добиваем тишиной, иначе поедет выравнивание.
                frame = bytes(self._tts).ljust(FRAME_BYTES, b"\x00")
                self._tts.clear()
            empty = not self._tts

        if empty:
            self._loop.call_soon_threadsafe(self.drained.set)
        return frame

    def _take_music(self) -> bytes | None:
        with self._music_lock:
            source, paused = self._music, self._music_paused
        if source is None or paused:
            return None

        data = source.read()
        if not data:
            self.set_music(None)
            if self._on_music_end is not None:
                self._loop.call_soon_threadsafe(self._on_music_end)
            return None
        if len(data) < FRAME_BYTES:
            data = data.ljust(FRAME_BYTES, b"\x00")
        return data

    def cleanup(self) -> None:
        self.set_music(None)
        with self._tts_lock:
            self._tts.clear()


class OutputArbiter:
    """Фасад над микшером: сериализует реплики бота."""

    def __init__(self, cfg, loop: asyncio.AbstractEventLoop) -> None:  # noqa: ANN001
        music = cfg.music
        self.source = MixerSource(
            loop=loop,
            volume=float(music.get("volume", 0.5)),
            duck_volume=float(music.get("duck_volume", 0.25)),
            fade_ms=int(music.get("duck_fade_ms", 150)),
        )
        # Бот говорит одним голосом: две реплики одновременно смешались бы в кашу.
        self._lock = asyncio.Lock()

    def bind_music_end(self, callback: Callable[[], None]) -> None:
        self.source._on_music_end = callback

    async def speak(self, chunks: AsyncIterator[tuple[bytes, int]]) -> None:
        """Проиграть поток кусков TTS и дождаться, пока всё прозвучит."""
        async with self._lock:
            spoke = False
            async for pcm, rate in chunks:
                mono = np.frombuffer(pcm, dtype="<i2")
                self.source.push_tts(mono_to_discord_pcm(mono, rate))
                spoke = True
            if spoke:
                await self.source.drained.wait()

    def interrupt(self) -> None:
        """Заткнуться немедленно (например, при выходе из канала)."""
        self.source.clear_tts()
