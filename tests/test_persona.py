"""Промпт: системная часть плюс ровно одна реплика, без истории и фона."""

from __future__ import annotations

from bashmak.llm.persona import SYSTEM_PROMPT, build_messages


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
