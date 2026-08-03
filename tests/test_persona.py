"""Промпт: системная часть плюс ровно одна реплика, без истории и фона.

И обратная сторона — разбор ответа: модель может уехать в дописывание диалога
за собеседника, и озвучивать это нельзя.
"""

from __future__ import annotations

import pytest

from bashmak.llm.persona import SYSTEM_PROMPT, build_messages, clean_reply


def test_prompt_is_system_plus_single_turn():
    messages = build_messages("Вася", "как дела")

    assert [m["role"] for m in messages] == ["system", "user"]
    assert messages[0]["content"] == SYSTEM_PROMPT
    assert messages[1]["content"] == "Вася: как дела"


def test_nothing_carries_over_between_questions():
    """Контекст сбрасывается: прошлый вопрос не должен просачиваться в новый."""
    build_messages("Вася", "первый вопрос")
    joined = " ".join(m["content"] for m in build_messages("Петя", "второй вопрос"))

    assert "первый вопрос" not in joined


def test_reply_is_cut_at_someone_elses_turn():
    """Живой случай: без разделителей модель дописала диалог за Васю."""
    raw = (
        ": Всё классно, а у тебя? Башмак всегда в хорошем настроении!"
        "userВася: у меня всё норм, спасибо. Башмак, ты знаешь, что такое «бодишейм»?"
        "availableassistant: Бодишейм — это когда стараешься хорошо выглядеть"
    )
    assert clean_reply(raw) == "Всё классно, а у тебя? Башмак всегда в хорошем настроении!"


@pytest.mark.parametrize(
    ("raw", "want"),
    [
        ("Норм.<|message_sep|>user<|role_sep|>Вася: ещё", "Норм."),
        ("Привет!available functions<|role_sep|>[]", "Привет!"),
        ("Да ну тебя.assistant: и ещё", "Да ну тебя."),
        (": ответ после пустого разделителя", "ответ после пустого разделителя"),
    ],
)
def test_foreign_markup_is_stripped(raw, want):
    assert clean_reply(raw) == want


@pytest.mark.parametrize(
    "reply",
    [
        "Иди нахуй.",
        "Спроси у Патриота, он знает.",
        "Не знаю, я в интернет не хожу.",
        "Два по цене одного: бери и не думай.",
    ],
)
def test_normal_reply_survives_untouched(reply):
    """Обрезка не должна кусать обычную речь — в том числе с двоеточием."""
    assert clean_reply(reply) == reply
