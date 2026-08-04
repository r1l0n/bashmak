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
        ("поставь музыку на паузу", Intent.MUSIC_PAUSE),
        ("следующий трек", Intent.MUSIC_SKIP),
        ("сделай музыку погромче", Intent.MUSIC_LOUDER),
        ("сделай песню потише", Intent.MUSIC_QUIETER),
    ],
)
def test_rules_catch_music_commands(text, intent):
    decision = classify_by_rules(text)
    assert decision is not None, f"правила не поймали {text!r}"
    assert decision.intent is intent
    assert decision.source == "regex"


@pytest.mark.parametrize(
    "text, intent",
    [
        ("поставь на паузу", Intent.MUSIC_PAUSE),
        ("продолжай", Intent.MUSIC_RESUME),
        ("давай дальше", Intent.MUSIC_SKIP),
        ("сделай погромче", Intent.MUSIC_LOUDER),
        ("потише пожалуйста", Intent.MUSIC_QUIETER),
    ],
)
def test_ambiguous_commands_need_playing_music(text, intent):
    """Без слова «музыка» это команды, только пока что-то играет."""
    decision = classify_by_rules(text, music_playing=True)
    assert decision is not None, f"правила не поймали {text!r}"
    assert decision.intent is intent


@pytest.mark.parametrize(
    "text",
    [
        # Всё это обычная речь, а не команды плееру.
        "продолжай",
        "а что там дальше",
        "говори потише",
        "давай погромче рассказывай",
        "сделай паузу в рассказе",
    ],
)
def test_ambiguous_words_are_chat_when_nothing_plays(text):
    assert classify_by_rules(text, music_playing=False) is None


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


@pytest.mark.parametrize(
    "text",
    [
        "завали ебало",
        "завали хлебало",
        "заткнись",
        "заткни ебало",
        "замолчи",
        "помолчи",
        "молчать",
        "заглохни",
        "тишина",
        "уймись",
        "стоп",
        "хватит болтать",
    ],
)
@pytest.mark.parametrize("music_playing", [False, True])
def test_silence_is_caught_with_or_without_music(text, music_playing):
    """Заткнуться бот обязан и когда музыки нет — это команда не плееру."""
    decision = classify_by_rules(text, music_playing=music_playing)
    assert decision is not None, f"правила не поймали {text!r}"
    assert decision.intent is Intent.SILENCE


@pytest.mark.parametrize(
    "text",
    [
        # Соседи по смыслу, которые новое правило не должно съесть.
        "сделай потише",
        "выключи музыку",
        "останови музыку",
        "поставь музыку на паузу",
        # Обычная речь с похожими корнями.
        "я завалил экзамен",
        "что такое молчание",
    ],
)
def test_silence_does_not_eat_neighbours(text):
    decision = classify_by_rules(text, music_playing=True)
    assert decision is None or decision.intent is not Intent.SILENCE


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
