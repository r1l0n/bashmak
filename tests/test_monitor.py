"""Монитор: разбор метрик и /proc.

Отрисовку не проверяем — она про глаза. Проверяем то, что может тихо
соврать: статистику, проценты загрузки и устойчивость к обрезанной строке.
"""

from __future__ import annotations

import json

import pytest

from bashmak import monitor


def _turn(total: float, **stages: float) -> str:
    return json.dumps({"at": 0, "speaker": "Вася", "stages": stages, "total": total})


def test_broken_line_is_skipped_not_fatal():
    """Монитор читает файл, пока бот в него пишет, — обрывок строки норма."""
    lines = ['{"at": 0, "stages": {"llm": 1.0}, "total": 1.0}', '{"at": 0, "sta']

    turns = monitor.parse_turns(lines)

    assert len(turns) == 1


def test_lines_without_stages_are_ignored():
    assert monitor.parse_turns(['{"что-то": "чужое"}', "", "   "]) == []


def test_percentiles_ignore_the_mean():
    """Одна долгая реплика не должна портить медиану."""
    p50, p90, top = monitor.percentiles([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 60.0])

    assert p50 == 1.0
    assert top == 60.0
    assert p90 <= 60.0


def test_percentiles_of_nothing_are_zeroes():
    assert monitor.percentiles([]) == (0.0, 0.0, 0.0)


def test_stage_stats_keep_pipeline_order():
    turns = monitor.parse_turns(
        [_turn(3.0, tts=0.5, llm=2.0, stt=0.5), _turn(4.0, tts=0.5, llm=3.0, stt=0.5)]
    )

    assert list(monitor.stage_stats(turns)) == ["stt", "llm", "tts", "всего"]


def test_stage_stats_include_total_from_the_turn():
    stats = monitor.stage_stats(monitor.parse_turns([_turn(9.0, llm=8.0)]))

    assert stats["всего"][0] == 9.0


def test_meminfo_uses_available_not_free():
    text = "MemTotal:       41943040 kB\nMemFree:         1048576 kB\nMemAvailable:   20971520 kB\n"

    used, total = monitor.parse_meminfo(text)

    assert total == pytest.approx(40.0)
    assert used == pytest.approx(20.0), "занято считаем по MemAvailable, иначе кеш выглядит утечкой"


def test_meminfo_falls_back_to_free():
    used, total = monitor.parse_meminfo("MemTotal: 41943040 kB\nMemFree: 41943040 kB\n")

    assert used == pytest.approx(0.0)


def test_cpu_percent_between_two_samples():
    first = monitor.parse_cpu_times("cpu  100 0 100 800 0 0 0 0\n")
    second = monitor.parse_cpu_times("cpu  200 0 200 1600 0 0 0 0\n")

    assert monitor.cpu_percent(first, second) == pytest.approx(20.0)


def test_cpu_percent_without_movement_is_zero():
    sample = monitor.parse_cpu_times("cpu  100 0 100 800 0 0 0 0\n")

    assert monitor.cpu_percent(sample, sample) == 0.0


def test_cpu_percent_survives_a_counter_reset():
    """После перезагрузки счётчики уезжают назад — процент не должен уйти в минус."""
    assert monitor.cpu_percent((900.0, 1000.0), (10.0, 20.0)) == 0.0


def test_sparkline_scales_to_the_maximum():
    line = monitor.sparkline([0.0, 5.0, 10.0], width=3)

    assert line[0] == "▁" and line[-1] == "█"


def test_sparkline_of_flat_zeroes_does_not_divide_by_zero():
    assert monitor.sparkline([0.0, 0.0]) == "▁▁"


def test_sparkline_of_nothing_is_empty():
    assert monitor.sparkline([]) == ""


def test_levels_missing_file_is_not_fatal(tmp_path):
    assert monitor.read_levels(tmp_path / "levels.json") == {}


def test_levels_half_written_file_is_not_fatal(tmp_path):
    path = tmp_path / "levels.json"
    path.write_text('{"users": [{"name": "Ва', encoding="utf-8")

    assert monitor.read_levels(path) == {}


def test_levels_round_trip(tmp_path):
    """Стык бота и монитора: что бот опубликовал, то монитор и прочитал."""
    from bashmak.utils.logging import publish_levels

    path = tmp_path / "levels.json"
    payload = {
        "at": 1.0,
        "threshold": 0.5,
        "users": [{"name": "балбес", "peak": 0.8, "vad": 0.97, "speech": True}],
    }

    publish_levels(path, payload)

    assert monitor.read_levels(path) == payload
    assert not (tmp_path / "levels.tmp").exists(), "временный файл должен быть переименован"


def test_bar_is_proportional_and_safe():
    assert monitor.bar(1.0, 2.0, width=10) == "█" * 5
    assert monitor.bar(1.0, 0.0) == ""
    assert monitor.bar(0.0, 5.0) == ""
