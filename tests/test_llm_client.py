"""Промпт для llama-server: форма messages и сборка формата GigaChat.

Формат собираем сами и шлём в /completion — чат-слой llama.cpp на нём
ломается (подробности в client.render_prompt). Значит, за корректность
разделителей теперь отвечаем мы, и проверять её нужно здесь.
"""

from __future__ import annotations

import asyncio
import json

import httpx
import pytest

from bashmak.llm.client import (
    MESSAGE_SEP,
    LlmClient,
    LlmError,
    _answer,
    _server_error,
    render_prompt,
)
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


# ------------------------------------------ обрыв на конце предложения ----
#
# Генерацию сверх первого предложения всё равно выбрасывает clean_reply, но
# считает её модель, и на CPU это секунды. Обрываем на сервере — а знак
# препинания, который сервер при этом съедает, возвращаем обратно.


@pytest.mark.parametrize(
    ("content", "stopping_word", "want"),
    [
        # Сервер оборвал на «! » и сам знак в content не положил.
        ("Иди нахер", "! ", "Иди нахер!"),
        # Другие сборки отдают знак уже без пробела.
        ("Не знаю", ".", "Не знаю."),
        ("Не знаю", "… ", "Не знаю…"),
        # Сборка без stopping_word: остаёмся без знака, но не падаем.
        ("Норм", None, "Норм"),
        # Оборвано концом реплики, а не предложения — дописывать нечего.
        ("Норм.", MESSAGE_SEP, "Норм."),
        # Пустой ответ знаком препинания не становится.
        ("", "! ", ""),
    ],
)
def test_sentence_mark_is_restored(content, stopping_word, want):
    data = {"content": content}
    if stopping_word is not None:
        data["stopping_word"] = stopping_word
    assert _answer(data) == want


def _capture():
    """Клиент с подменённым транспортом: запоминает, что ушло на сервер."""
    sent: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        sent.update(json.loads(request.content))
        return httpx.Response(200, json={"content": "Ответ", "stopping_word": ""})

    client = LlmClient(FakeSection())
    client._client = httpx.AsyncClient(
        transport=httpx.MockTransport(handler), base_url="http://llm.test"
    )
    return client, sent


def _ask(client, messages, **kwargs) -> str:
    async def main():
        try:
            return await client.complete(messages, **kwargs)
        finally:
            await client.close()

    return asyncio.run(main())


class FakeSection:
    """Мини-заглушка Config.Section: клиент читает секцию только через get()."""

    def __init__(self, **values):
        self._values = values

    def get(self, name, default=None):
        return self._values.get(name, default)


def test_chat_request_stops_at_the_first_sentence():
    client, sent = _capture()

    _ask(client, build_messages("Вася", "как дела"), one_sentence=True)

    assert MESSAGE_SEP in sent["stop"]
    assert {". ", "! ", "? ", "… "} <= set(sent["stop"])


def test_other_requests_keep_the_full_output():
    """Стоп по «. » порезал бы JSON классификатора на «Кино. Группа крови»."""
    client, sent = _capture()

    _ask(client, [{"role": "user", "content": "включи Кино"}])

    assert sent["stop"] == [MESSAGE_SEP]
