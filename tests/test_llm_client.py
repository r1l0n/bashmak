"""Проверка формы messages перед отправкой в llama-server.

Шаблон GigaChat отвергает всё, кроме строгого чередования user/assistant
после system. Сломать это легко (склеить историю, забыть роль), а увидеть
трудно: ошибка приходит с сервера как 500.
"""

from __future__ import annotations

import httpx
import pytest

from bashmak.llm.client import LlmClient, LlmError, _server_error
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


def test_server_error_keeps_the_explanation():
    """Ровно то, чего не хватило на 500: причина, а не только код."""
    reason = "Failed to apply chat template: Unknown statement at row 3"
    response = httpx.Response(500, json={"error": {"message": reason, "code": 500}})

    message = _server_error(response)

    assert reason in message
    assert "500" in message


def test_server_error_survives_a_plain_text_body():
    assert "loading model" in _server_error(httpx.Response(503, text="loading model"))


def test_server_error_says_so_when_body_is_empty():
    assert "пустой" in _server_error(httpx.Response(500, text=""))
