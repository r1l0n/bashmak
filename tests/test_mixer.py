"""Микшер голосового выхода: приоритет речи, даккинг музыки, отсутствие клиппинга."""

from __future__ import annotations

import asyncio

import numpy as np
import pytest

from bashmak.output import arbiter as arbiter_module
from bashmak.output.arbiter import (
    FRAME_BYTES,
    FRAME_SAMPLES,
    SILENCE,
    MixerSource,
    OutputArbiter,
)


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
    mixer = MixerSource(loop=loop, on_music_end=ended.append)
    source = FakeMusic([frame(100)])
    mixer.set_music(source)

    mixer.read()  # единственный кадр
    assert mixer.read() == SILENCE  # источник иссяк
    assert not mixer.has_music

    # И колбэк, и cleanup() уходят в луп через call_soon_threadsafe:
    # cleanup() у ffmpeg блокирующий, а поток плеера этого не переживёт.
    assert not source.cleaned, "cleanup не должен вызываться из аудио-потока"
    loop.call_soon(loop.stop)
    loop.run_forever()

    assert source.cleaned
    # Колбэк получает сам источник — иначе плеер не отличит «мой трек кончился»
    # от «кончился тот, что я уже снял».
    assert ended == [source]


def test_stale_track_end_is_distinguishable(loop):
    """Трек кончился ровно в момент skip(): колбэк должен нести старый источник."""
    ended = []
    mixer = MixerSource(loop=loop, on_music_end=ended.append)
    finished = FakeMusic([])
    mixer.set_music(finished)

    mixer.read()  # источник сразу пуст → запланирован колбэк
    replacement = FakeMusic([frame(100)])
    mixer.set_music(replacement)

    loop.call_soon(loop.stop)
    loop.run_forever()

    assert ended == [finished]
    assert ended[0] is not replacement


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


def test_ducking_stays_relative_on_low_volume(loop):
    """Громкость ниже duck_volume не должна инвертировать даккинг."""
    mixer = MixerSource(loop=loop, volume=0.5, duck_volume=0.25, fade_ms=20)
    mixer.set_volume(0.2)  # две команды «потише» — тише абсолютного порога

    payload = frame(10000)
    mixer.set_music(FakeMusic([payload, payload]))

    idle = np.frombuffer(mixer.read(), dtype="<i2").max()
    mixer.push_tts(frame(1))
    ducked = np.frombuffer(mixer.read(), dtype="<i2").max()

    assert ducked < idle, "пока бот говорит, музыка обязана становиться тише"


def test_drained_is_not_set_while_next_chunk_arrives(loop):
    """Гонка на drained: отложенный set() не должен отпускать непустой буфер."""
    mixer = MixerSource(loop=loop)
    mixer.push_tts(frame(100))

    mixer.read()  # буфер опустел → set() запланирован в луп
    mixer.push_tts(frame(200))  # но приехал следующий кусок той же реплики

    loop.call_soon(loop.stop)
    loop.run_forever()

    assert not mixer.drained.is_set()
    assert mixer.speaking


class FakeConfig:
    """Мини-заглушка Config: арбитру нужна только секция music."""

    class _Music:
        @staticmethod
        def get(name, default=None):
            return default

    music = _Music()


def test_speak_gives_up_when_player_is_dead(loop, monkeypatch):
    """Голосовое соединение отвалилось — speak() обязан отпустить очередь LLM."""
    monkeypatch.setattr(arbiter_module, "SPEAK_TIMEOUT_SLACK", 0.05)
    out = OutputArbiter(FakeConfig(), loop)

    async def chunks():
        yield np.full(480, 1000, dtype="<i2").tobytes(), 48000

    # read() никто не зовёт: плеера нет, drained не наступит никогда.
    loop.run_until_complete(asyncio.wait_for(out.speak(chunks()), 5.0))

    assert not out.source.speaking
