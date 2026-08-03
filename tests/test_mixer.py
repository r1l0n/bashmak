"""Микшер голосового выхода: приоритет речи, даккинг музыки, отсутствие клиппинга."""

from __future__ import annotations

import asyncio

import numpy as np
import pytest

from bashmak.output.arbiter import FRAME_BYTES, FRAME_SAMPLES, SILENCE, MixerSource


@pytest.fixture()
def loop():
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


def frame(value: int) -> bytes:
    """Кадр 20 мс: 960 сэмплов × 2 канала, одно и то же значение."""
    return np.full(FRAME_SAMPLES * 2, value, dtype="<i2").tobytes()


class FakeMusic:
    def __init__(self, frames):
        self.frames = list(frames)
        self.cleaned = False

    def read(self) -> bytes:
        return self.frames.pop(0) if self.frames else b""

    def cleanup(self) -> None:
        self.cleaned = True


def test_idle_mixer_emits_silence_forever(loop):
    """Пустой read() вернул бы b'' и навсегда убил бы плеер — так нельзя."""
    mixer = MixerSource(loop=loop)
    for _ in range(3):
        data = mixer.read()
        assert data == SILENCE
        assert len(data) == FRAME_BYTES


def test_speech_passes_through_untouched(loop):
    mixer = MixerSource(loop=loop)
    payload = frame(1000)
    mixer.push_tts(payload)

    assert mixer.speaking
    assert mixer.read() == payload
    assert not mixer.speaking
    assert mixer.read() == SILENCE


def test_short_tail_is_padded_to_full_frame(loop):
    mixer = MixerSource(loop=loop)
    mixer.push_tts(frame(500)[: FRAME_BYTES // 2])

    data = mixer.read()
    assert len(data) == FRAME_BYTES
    assert data[FRAME_BYTES // 2 :] == bytes(FRAME_BYTES // 2)


def test_music_alone_plays_at_full_volume(loop):
    mixer = MixerSource(loop=loop, volume=1.0)
    payload = frame(800)
    mixer.set_music(FakeMusic([payload]))

    assert mixer.read() == payload


def test_music_is_ducked_while_speaking(loop):
    mixer = MixerSource(loop=loop, volume=1.0, duck_volume=0.0, fade_ms=20)
    speech = frame(1000)
    mixer.push_tts(speech)
    mixer.set_music(FakeMusic([frame(9000), frame(9000)]))

    # Фейд рассчитан на один кадр — музыка уходит в ноль сразу.
    assert mixer.read() == speech

    # Речь кончилась — музыка возвращается.
    restored = np.frombuffer(mixer.read(), dtype="<i2")
    assert restored.max() > 0


def test_finished_track_is_released_and_reported(loop):
    ended = []
    mixer = MixerSource(loop=loop, on_music_end=lambda: ended.append(True))
    source = FakeMusic([frame(100)])
    mixer.set_music(source)

    mixer.read()  # единственный кадр
    assert mixer.read() == SILENCE  # источник иссяк
    assert not mixer.has_music
    assert source.cleaned

    # Колбэк уходит в луп через call_soon_threadsafe — прокрутим луп разок.
    loop.call_soon(loop.stop)
    loop.run_forever()
    assert ended == [True]


def test_mix_does_not_overflow_int16(loop):
    mixer = MixerSource(loop=loop, volume=1.0, duck_volume=1.0, fade_ms=20)
    mixer.push_tts(frame(30000))
    mixer.set_music(FakeMusic([frame(30000)]))

    mixed = np.frombuffer(mixer.read(), dtype="<i2")
    assert mixed.max() == 32767
    assert mixed.min() >= -32768


def test_paused_music_is_not_consumed(loop):
    mixer = MixerSource(loop=loop, volume=1.0)
    source = FakeMusic([frame(700), frame(700)])
    mixer.set_music(source)
    mixer.pause_music()

    assert mixer.read() == SILENCE
    assert len(source.frames) == 2, "на паузе трек не должен проматываться"

    mixer.resume_music()
    assert mixer.read() == frame(700)


def test_volume_is_clamped(loop):
    mixer = MixerSource(loop=loop, volume=1.0)
    assert mixer.set_volume(5.0) == 1.5
    assert mixer.set_volume(-1.0) == 0.0
