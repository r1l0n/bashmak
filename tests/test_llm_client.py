"""Проверка формы messages перед отправкой в llama-server.

Шаблон GigaChat отвергает всё, кроме строгого чередования user/assistant
после system. Сломать это легко (склеить историю, забыть роль), а увидеть
трудно: ошибка приходит с сервера как 500.
"""

from __future__ import annotations

import pytest

from bashmak.llm.client import LlmClient, LlmError
from bashmak.llm.persona import build_messages


def test_persona_builds_an_acceptable_shape():
    LlmClient._check_roles(build_messages("Вася", "как дела"))


def test_alternating_history_is_accepted():
    LlmClient._check_roles(
        [
            {"role": "system", "content": "s"},
            {"role": "user", "content": "1"},
            {"role": "assistant", "content": "2"},
            {"role": "user", "content": "3"},
        ]
    )


def test_two_user_messages_in_a_row_are_rejected():
    with pytest.raises(LlmError):
        LlmClient._check_roles(
            [
                {"role": "system", "content": "s"},
                {"role": "user", "content": "1"},
                {"role": "user", "content": "2"},
            ]
        )
