"""Единственный голосовой выход канала.

В канал физически уходит один поток, а претендентов на него два: ответы бота
и музыка. ТЗ предлагает ставить музыку на паузу, но в discord.py это выходит
криво: ``voice_client`` держит ровно один источник, вызвать ``play()`` поверх
паузы нельзя, а значит пришлось бы останавливать трек и пересоздавать
``FFmpegPCMAudio`` с ``-ss <позиция>`` — то есть рывок, потеря позиции при
живом стриме и лишний коннект к YouTube на каждую реплику.

Поэтому источник ровно один и на всё время: :class:`MixerSource`. Он сам
складывает музыку с речью и приглушает музыку, пока бот говорит. Ничего не
перезапускается, приоритет TTS соблюдён, позиция трека не теряется.

``read()`` вызывается плеером discord.py из отдельного потока каждые 20 мс —
всё, что тут происходит, обязано быть арифметикой над 960 сэмплами.
"""

from __future__ import annotations

import asyncio
import logging
import threading
from typing import AsyncIterator, Callable

import discord
import numpy as np

from ..audio.resample import (
    DISCORD_RATE,
    StreamResampler,
    float_mono_to_discord_pcm,
    to_float_mono,
)

log = logging.getLogger(__name__)

#: 20 мс при 48 кГц стерео int16 — размер кадра, который ждёт discord.py.
FRAME_SAMPLES = 960
FRAME_BYTES = FRAME_SAMPLES * 2 * 2
SILENCE = bytes(FRAME_BYTES)
#: Сколько байт микса уходит в канал за секунду реального времени.
BYTES_PER_SECOND = FRAME_BYTES * 50
#: Запас к расчётной длительности реплики, прежде чем считать выход мёртвым.
SPEAK_TIMEOUT_SLACK = 10.0


class MixerSource(discord.AudioSource):
    """Микс «музыка + речь» с даккингом. Живёт, пока бот в канале."""

    def __init__(
        self,
        *,
        loop: asyncio.AbstractEventLoop,
        volume: float = 0.5,
        duck_volume: float = 0.25,
        fade_ms: int = 150,
        on_music_end: Callable[[discord.AudioSource], None] | None = None,
    ) -> None:
        self._loop = loop
        self._on_music_end = on_music_end

        self.volume = float(volume)
        self.duck_volume = float(duck_volume)
        # Даккинг относительный: «приглушить до четверти» должно означать
        # четверть от ТЕКУЩЕЙ громкости, а не абсолютные 0.25. Иначе после
        # пары команд «потише» музыка становилась бы громче, пока бот говорит.
        base = self.volume if self.volume > 0 else 1.0
        self._duck_ratio = min(1.0, max(0.0, self.duck_volume / base))
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

    def _to_loop(self, callback: Callable[..., None], *args) -> None:
        """Выполнить в лупе. На закрытом лупе (выключение) — на месте."""
        try:
            self._loop.call_soon_threadsafe(callback, *args)
        except RuntimeError:
            callback(*args)

    # ------------------------------------------------------------ речь ----
    def push_tts(self, pcm: bytes) -> None:
        """Добавить синтезированный кусок. Вызывается из лупа."""
        if not pcm:
            return
        # clear() строго под тем же локом, что и extend: иначе поток плеера
        # успевает между ними опустошить буфер и запланировать set(), который
        # выполнится уже после clear() — и speak() отпустит реплику раньше
        # времени, наложив следующую на недоигранную.
        with self._tts_lock:
            self._tts.extend(pcm)
            self.drained.clear()

    def clear_tts(self) -> None:
        with self._tts_lock:
            self._tts.clear()
        self._to_loop(self._sync_drained)

    def _sync_drained(self) -> None:
        """Выставить событие, только если буфер и правда пуст. Только из лупа."""
        with self._tts_lock:
            empty = not self._tts
        if empty:
            self.drained.set()

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
            # cleanup() у FFmpegPCMAudio — это kill() + communicate(), то есть
            # блокирующая операция. Вызывать её прямо тут нельзя: set_music
            # дёргается в том числе из потока плеера, у которого на кадр 20 мс.
            self._to_loop(self._release, old)

    @staticmethod
    def _release(source: discord.AudioSource) -> None:
        try:
            source.cleanup()
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

        # Приглушение — доля от текущей громкости, а не абсолютная величина.
        duck = self._duck_ratio
        music_gain = self.volume * (duck + (1.0 - duck) * self._gain_ratio)

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
            self._to_loop(self._sync_drained)
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
                # Передаём сам источник: пока колбэк доедет до лупа, плеер уже
                # мог запустить следующий трек, и «трек кончился» относилось бы
                # не к тому источнику.
                self._to_loop(self._on_music_end, source)
            return None
        if len(data) < FRAME_BYTES:
            data = data.ljust(FRAME_BYTES, b"\x00")
        return data

    def cleanup(self) -> None:
        self.set_music(None)
        self.clear_tts()


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

    def bind_music_end(self, callback: Callable[[discord.AudioSource], None]) -> None:
        self.source._on_music_end = callback

    async def speak(self, chunks: AsyncIterator[tuple[bytes, int]]) -> None:
        """Проиграть поток кусков TTS и дождаться, пока всё прозвучит."""
        async with self._lock:
            pushed = 0
            # Один ресемплер на реплику: покадровый stateless-пересчёт даёт
            # щелчки на стыках предложений.
            resampler: StreamResampler | None = None

            async for pcm, rate in chunks:
                if resampler is None or resampler.in_rate != rate:
                    if resampler is not None:
                        pushed += self._push(resampler.flush())
                    resampler = StreamResampler(rate, DISCORD_RATE)
                mono = to_float_mono(np.frombuffer(pcm, dtype="<i2"))
                pushed += self._push(resampler.process(mono))

            if resampler is not None:
                pushed += self._push(resampler.flush())

            if not pushed:
                return

            # Ждать событие без таймаута нельзя: его выставляет поток плеера, а
            # он крутится, только пока живо голосовое соединение. Оборвалось —
            # и вся очередь LLM встала бы намертво до рестарта процесса.
            timeout = pushed / BYTES_PER_SECOND + SPEAK_TIMEOUT_SLACK
            try:
                await asyncio.wait_for(self.source.drained.wait(), timeout)
            except asyncio.TimeoutError:
                log.warning(
                    "речь не доиграла за %.0f с — голосовой выход не крутится, "
                    "сбрасываю буфер",
                    timeout,
                )
                self.source.clear_tts()

    def _push(self, samples: np.ndarray) -> int:
        data = float_mono_to_discord_pcm(samples)
        self.source.push_tts(data)
        return len(data)

    def interrupt(self) -> None:
        """Заткнуться немедленно (например, при выходе из канала)."""
        self.source.clear_tts()
