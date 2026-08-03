"""Автомат нарезки речи. Саму модель Silero подменяем сценарием вероятностей."""

from __future__ import annotations

import numpy as np
import pytest

from bashmak.audio.vad import Segmenter

WINDOW = 512
RATE = 16000


class ScriptedVad:
    """Отдаёт заранее заданные вероятности вместо инференса."""

    sample_rate = RATE

    def __init__(self, script):
        self.script = list(script)
        self.calls = 0
        self.resets = 0

    def speech_probability(self, window):
        self.calls += 1
        return self.script.pop(0) if self.script else 0.0

    def reset(self):
        self.resets += 1


def make_segmenter(script, **overrides):
    params = dict(
        threshold=0.5,
        silence_ms=96,  # 3 окна по 32 мс
        min_speech_ms=64,  # 2 окна
        max_segment_s=0.32,  # 10 окон
        preroll_ms=64,  # 2 окна + текущее
        window_samples=WINDOW,
    )
    params.update(overrides)
    return Segmenter(1, ScriptedVad(script), **params)


def audio(n_windows: int) -> np.ndarray:
    return np.zeros(n_windows * WINDOW, dtype=np.float32)


def test_speech_between_silences_becomes_one_segment():
    script = [0.0, 0.0] + [0.9] * 5 + [0.0] * 3
    segmenter = make_segmenter(script)

    segments = segmenter.feed(audio(len(script)))

    assert len(segments) == 1
    assert segments[0].user_id == 1
    # Пре-ролл добавлен, хвост тишины срезан.
    assert 0.15 < segments[0].duration < 0.35
    assert segmenter.vad.resets == 1
    assert not segmenter.in_speech


def test_too_short_burst_is_dropped():
    """Одиночный щелчок короче min_speech_ms — не фраза."""
    script = [0.0, 0.0, 0.9, 0.0, 0.0, 0.0]
    segmenter = make_segmenter(script)

    assert segmenter.feed(audio(len(script))) == []


def test_long_monologue_is_force_split():
    script = [0.9] * 25
    segmenter = make_segmenter(script)

    segments = segmenter.feed(audio(len(script)))

    assert len(segments) >= 2, "монолог должен резаться по max_segment_s"


def test_flush_closes_unfinished_phrase():
    script = [0.0, 0.9, 0.9, 0.9]
    segmenter = make_segmenter(script)

    assert segmenter.feed(audio(len(script))) == []
    assert segmenter.in_speech

    tail = segmenter.flush()
    assert tail is not None
    assert tail.duration > 0
    assert segmenter.flush() is None


def test_partial_window_is_buffered_not_lost():
    """Кадры Discord не кратны окну VAD — остаток должен доживать до следующего вызова."""
    segmenter = make_segmenter([0.0] * 10)

    segmenter.feed(np.zeros(WINDOW - 10, dtype=np.float32))
    assert segmenter.vad.calls == 0

    segmenter.feed(np.zeros(10, dtype=np.float32))
    assert segmenter.vad.calls == 1


def test_silence_only_produces_nothing():
    segmenter = make_segmenter([0.0] * 20)
    assert segmenter.feed(audio(20)) == []
    assert not segmenter.in_speech


@pytest.mark.parametrize("window_samples", [512])
def test_window_size_defines_timing(window_samples):
    segmenter = make_segmenter([0.0], window_samples=window_samples)
    assert segmenter.window_ms == pytest.approx(window_samples * 1000 / RATE)
