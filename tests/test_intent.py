"""Быстрый путь intent-роутера: регексы без LLM."""

from __future__ import annotations

import pytest

from bashmak.intent.router import Intent, classify_by_rules, extract_level, extract_query


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
    """Команда молчания срабатывает и без музыки: она адресована не плееру."""
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


@pytest.mark.parametrize(
    "text, level",
    [
        # Ровно то, что было в логе: раньше это уезжало в music_louder.
        ("громкость 100", 100),
        ("громкость 50", 50),
        ("поставь громкость на 30", 30),
        ("сделай громкость 70 процентов", 70),
        ("громкость до 20", 20),
        ("выстави громкость в 0", 0),
        # Whisper пишет числа то цифрами, то прописью.
        ("громкость сто", 100),
        ("громкость пятьдесят", 50),
        ("громкость на восемьдесят", 80),
    ],
)
@pytest.mark.parametrize("music_playing", [False, True])
def test_volume_level_is_set_not_stepped(text, level, music_playing):
    """Названо число — это установка значения, а не шаг «погромче»."""
    decision = classify_by_rules(text, music_playing=music_playing)
    assert decision is not None, f"правила не поймали {text!r}"
    assert decision.intent is Intent.MUSIC_VOLUME
    assert decision.level == level


@pytest.mark.parametrize(
    "text, intent",
    [
        # Без числа это по-прежнему шаг.
        ("сделай погромче", Intent.MUSIC_LOUDER),
        ("прибавь громкость", Intent.MUSIC_LOUDER),
        ("сделай потише", Intent.MUSIC_QUIETER),
        ("убавь громкость", Intent.MUSIC_QUIETER),
    ],
)
def test_volume_without_a_number_is_still_a_step(text, intent):
    decision = classify_by_rules(text, music_playing=True)
    assert decision is not None
    assert decision.intent is intent
    assert decision.level is None


@pytest.mark.parametrize(
    "text",
    [
        # Число рядом, но не про громкость — правило не должно на него прыгать.
        "включи звуки природы 10 часов",
        "поставь кино 1988",
        "включи трек на 5 минут",
    ],
)
def test_volume_rule_does_not_grab_stray_numbers(text):
    decision = classify_by_rules(text, music_playing=True)
    assert decision is not None
    assert decision.intent is Intent.MUSIC_PLAY


def test_extract_level_takes_a_bare_number_from_the_classifier():
    """LLM отдаёт число отдельным полем, без слова «громкость»."""
    assert extract_level("40") == 40
    assert extract_level("") is None
    assert extract_level("погромче") is None


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
        # «Любой» и «на свой вкус» — тоже шум: без этого бот ушёл бы искать на
        # YouTube слово «любой».
        ("включи любой трек", ""),
        ("поставь что-нибудь на свой вкус", ""),
        ("врубни какую-нибудь музыку", ""),
    ],
)
def test_extract_query_drops_filler(text, expected):
    assert extract_query(text) == expected


@pytest.mark.parametrize(
    "text",
    [
        "включи что-нибудь",
        "включи музыку",
        "поставь любую песню",
        "врубни что-нибудь на свой вкус",
        "поставь что-то",
        "включи какой-нибудь трек",
        "врубни рандомный трек",
        "включи чего-нибудь",
        "поставь музыку сам выбери",
    ],
)
def test_play_without_a_name_is_a_command_to_choose(text):
    """Пустой запрос — это «выбери сам», а не «команду не поняли»."""
    decision = classify_by_rules(text)
    assert decision is not None, f"правила не поймали {text!r}"
    assert decision.intent is Intent.MUSIC_PLAY
    assert decision.query == ""


@pytest.mark.parametrize(
    "text",
    [
        # «Что-нибудь» само по себе музыку не заказывает.
        "расскажи что-нибудь",
        "скажи что-нибудь смешное",
        "что-нибудь новенькое есть",
        "любой каприз за ваши деньги",
    ],
)
def test_choose_yourself_does_not_eat_chat(text):
    assert classify_by_rules(text) is None


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
