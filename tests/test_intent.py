"""Быстрый путь intent-роутера: регексы без LLM."""

from __future__ import annotations

import pytest

from bashmak.intent.router import Intent, classify_by_rules, extract_query


@pytest.mark.parametrize(
    "text, intent",
    [
        ("включи кино группа крови", Intent.MUSIC_PLAY),
        ("поставь что-нибудь из наутилуса", Intent.MUSIC_PLAY),
        ("выключи музыку", Intent.MUSIC_STOP),
        ("останови музыку", Intent.MUSIC_STOP),
        ("поставь на паузу", Intent.MUSIC_PAUSE),
        ("продолжай", Intent.MUSIC_RESUME),
        ("следующий трек", Intent.MUSIC_SKIP),
        ("давай дальше", Intent.MUSIC_SKIP),
        ("сделай погромче", Intent.MUSIC_LOUDER),
        ("потише пожалуйста", Intent.MUSIC_QUIETER),
    ],
)
def test_rules_catch_music_commands(text, intent):
    decision = classify_by_rules(text)
    assert decision is not None, f"правила не поймали {text!r}"
    assert decision.intent is intent
    assert decision.source == "regex"


@pytest.mark.parametrize(
    "text",
    [
        "расскажи анекдот",
        "как дела",
        "что ты думаешь про питон",
        # «включи» без объекта — начало обычной фразы, а не команда плееру.
        "включи",
    ],
)
def test_chat_falls_through_rules(text):
    assert classify_by_rules(text) is None


def test_stop_wins_over_play():
    """«выключи музыку» содержит и «ключ», и «музык» — важен порядок правил."""
    decision = classify_by_rules("выключи музыку пожалуйста")
    assert decision is not None
    assert decision.intent is Intent.MUSIC_STOP


@pytest.mark.parametrize(
    "text, expected",
    [
        ("включи нам что-нибудь из кино", "из кино"),
        ("поставь песню группы алиса", "группы алиса"),
        ("врубни фоном джаз", "джаз"),
    ],
)
def test_extract_query_drops_filler(text, expected):
    assert extract_query(text) == expected


def test_play_carries_query():
    decision = classify_by_rules("включи кино группа крови")
    assert decision is not None
    assert decision.intent is Intent.MUSIC_PLAY
    assert decision.query == "кино группа крови"


def test_play_without_query_is_still_a_music_command():
    """«включи музыку» — команда плееру, хоть название и не названо."""
    decision = classify_by_rules("включи музыку")
    assert decision is not None
    assert decision.intent is Intent.MUSIC_PLAY
    assert decision.query == ""
