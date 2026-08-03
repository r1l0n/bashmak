"""История диалога: фон не вытесняет разговор, роли не идут сериями."""

from __future__ import annotations

from bashmak.llm.persona import SYSTEM_PROMPT, Conversation


def test_ambient_lines_do_not_evict_dialogue():
    """Трое болтающих не должны стирать вопрос и ответ бота."""
    talk = Conversation(max_turns=4, ambient_lines=2)
    talk.add_user("Вася", "как тебя зовут")
    talk.add_assistant("Башмак")

    for index in range(20):
        talk.note_ambient("Петя", f"фоновая реплика {index}")

    messages = talk.build_messages("Вася", "а повтори")
    joined = " ".join(message["content"] for message in messages)

    assert "как тебя зовут" in joined
    assert "Башмак" in joined
    # Фона осталось ровно столько, сколько разрешено.
    assert "фоновая реплика 18" in joined
    assert "фоновая реплика 19" in joined
    assert "фоновая реплика 17" not in joined


def test_consecutive_user_turns_are_merged():
    talk = Conversation(max_turns=8, ambient_lines=2)
    talk.add_user("Вася", "первый вопрос")
    talk.add_assistant("первый ответ")
    talk.add_user("Вася", "второй вопрос")
    talk.note_ambient("Петя", "что-то своё")

    messages = talk.build_messages("Вася", "третий вопрос")
    roles = [message["role"] for message in messages]

    assert messages[0]["content"] == SYSTEM_PROMPT
    assert roles == ["system", "user", "assistant", "user"]
    assert "третий вопрос" in messages[-1]["content"]
    assert "что-то своё" in messages[-1]["content"]


def test_ambient_is_consumed_by_a_real_turn():
    talk = Conversation(max_turns=8, ambient_lines=4)
    talk.note_ambient("Петя", "мимо бота")
    talk.add_user("Вася", "вопрос")
    talk.add_assistant("ответ")

    joined = " ".join(m["content"] for m in talk.build_messages("Вася", "ещё"))
    assert "мимо бота" not in joined, "фон уже уехал в промпт, второй раз не нужен"
