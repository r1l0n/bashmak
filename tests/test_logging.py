"""Отчёт по реплике: что собирается, что печатается, что выбрасывается."""

from __future__ import annotations

import logging

import pytest

from bashmak.utils import logging as blog


@pytest.fixture(autouse=True)
def _fresh_turn(monkeypatch):
    """Каждому тесту — свой cid и чистое накопленное состояние."""
    monkeypatch.setattr(blog, "_open_turns", type(blog._open_turns)())
    monkeypatch.setattr(blog, "_recent", type(blog._recent)(maxlen=10))
    token = blog.current_cid.set("u1@1.0")
    yield
    blog.current_cid.reset(token)


def _record_stages(**stages: float) -> None:
    for name, seconds in stages.items():
        blog._record_stage(name, seconds)


def test_tts_chunks_collapse_into_one_stage():
    """Синтез идёт кусками (tts[0], tts[1]) — в отчёте это одна строка."""
    blog._record_stage("tts[0]", 0.4)
    blog._record_stage("tts[1]", 0.6)

    assert blog._open_turns["u1@1.0"].stages == {"tts": pytest.approx(1.0)}


def test_llm_intent_stage_is_named_like_the_others():
    blog._record_stage("intent-llm", 0.2)

    assert list(blog._open_turns["u1@1.0"].stages) == ["intent"]


def test_report_prints_heard_sent_and_reply(caplog):
    caplog.set_level(logging.INFO)
    _record_stages(stt=0.5, llm=2.0)
    blog.turn_note(speaker="балбес", heard="иди нахуй", sent="балбес: иди нахуй", reply="сам иди")

    blog.turn_report()

    text = caplog.text
    assert "балбес" in text
    assert "иди нахуй" in text and "сам иди" in text
    assert "stt 0.5" in text and "llm 2.0" in text
    assert blog._open_turns == {}, "отчёт напечатан — реплику надо закрыть"


def test_dropped_turn_prints_nothing(caplog):
    caplog.set_level(logging.INFO)
    _record_stages(stt=0.5)
    blog.turn_note(speaker="кто-то", heard="болтовня мимо бота")

    blog.turn_drop()
    blog.turn_report()

    assert "болтовня мимо бота" not in caplog.text
    assert blog._open_turns == {}


def test_report_without_a_turn_is_harmless():
    blog.turn_report()  # реплики не было — не должно падать


def test_summary_shows_a_bar_per_stage():
    _record_stages(stt=0.5, llm=2.0)
    blog.turn_report()

    summary = "\n".join(blog.stage_summary())

    assert "stt" in summary and "llm" in summary and "всего" in summary
    assert "█" in summary


def test_metrics_line_is_readable_by_the_monitor(tmp_path, monkeypatch):
    """Стык бота и монитора: что записали, то он и должен разобрать."""
    from bashmak import monitor

    path = tmp_path / "turns.jsonl"
    blog._setup_metrics(tmp_path / "bashmak.log", 1_000_000, 2)
    monkeypatch.setattr(blog, "METRICS_NAME", "turns.jsonl")

    _record_stages(stt=0.5, llm=2.0)
    blog.turn_note(speaker="балбес", heard="иди нахуй", reply="сам иди")
    blog.turn_report()
    for handler in blog._metrics.handlers:
        handler.flush()

    turns = monitor.read_turns(path)

    assert len(turns) == 1
    assert turns[0]["speaker"] == "балбес"
    assert turns[0]["stages"]["llm"] == pytest.approx(2.0)
    # total — это настоящее время жизни реплики; в тесте оно микросекундное.
    assert turns[0]["total"] >= 0


def test_metrics_without_a_log_file_do_not_crash():
    blog._setup_metrics(None, 1_000_000, 2)
    _record_stages(stt=0.1)
    blog.turn_note(speaker="Вася", heard="привет")

    blog.turn_report()  # писать некуда — и не надо


def test_summary_says_when_there_is_nothing_yet():
    assert blog.stage_summary() == ["замеров пока нет"]


def test_long_text_is_clipped():
    assert blog._clip("а" * 300).endswith("…")
    assert len(blog._clip("а" * 300)) == 160
    assert blog._clip("две   строки\nв одну") == "две строки в одну"


def test_bar_length_is_proportional():
    assert blog._bar(1.0, 2.0, width=10) == "█" * 5
    assert blog._bar(0.0, 2.0) == ""
    assert blog._bar(1.0, 0.0) == "", "нулевой итог не должен делить на ноль"
