"""Промпт для llama-server: форма messages и сборка формата GigaChat.

Формат собираем сами и шлём в /completion — чат-слой llama.cpp на нём
ломается (подробности в client.render_prompt). Значит, за корректность
разделителей теперь отвечаем мы, и проверять её нужно здесь.
"""

from __future__ import annotations

import httpx
import pytest

from bashmak.llm.client import LlmClient, LlmError, _server_error, render_prompt
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


def test_prompt_matches_the_reference_gigachat_format():
    """Эталон сверен с штатным шаблоном модели, отрисованным настоящим Jinja."""
    prompt = render_prompt(
        [{"role": "system", "content": "S"}, {"role": "user", "content": "V: q"}]
    )

    assert prompt == (
        "S<|message_sep|>"
        "user<|role_sep|>V: q<|message_sep|>"
        "available functions<|role_sep|>[]<|message_sep|>"
        "assistant<|role_sep|>"
    )


def test_prompt_has_no_bos_because_the_server_adds_it():
    """/completion токенизирует с add_special=true — второй BOS был бы лишним."""
    assert "<s>" not in render_prompt(build_messages("Вася", "как дела"))


def test_every_user_turn_gets_the_functions_block():
    prompt = render_prompt(
        [
            {"role": "system", "content": "S"},
            {"role": "user", "content": "1"},
            {"role": "assistant", "content": "2"},
            {"role": "user", "content": "3"},
        ]
    )

    assert prompt.count("available functions") == 2
    assert prompt.endswith("assistant<|role_sep|>")
    # Ответ ассистента идёт со своей ролью и без блока функций после неё.
    assert "assistant<|role_sep|>2<|message_sep|>user<|role_sep|>3" in prompt
