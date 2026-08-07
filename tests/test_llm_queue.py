"""Очередь LLM: короткая история на канал и команда молчания.

Клиент и озвучка подменены заглушками, проверяется только то, что делает сама
очередь: что уходит в модель, что доходит до озвучки и что остаётся в памяти.
"""

from __future__ import annotations

import asyncio
import time

import pytest

from bashmak.llm.queue_manager import ChatTask, LlmQueue

CHANNEL = 100


class FakeSection:
    """Мини-заглушка Config.Section: очередь читает секцию только через get()."""

    def __init__(self, **values):
        self._values = values

    def get(self, name, default=None):
        return self._values.get(name, default)


class FakeClient:
    """Отвечает по сценарию (или по счётчику) и запоминает, что ему прислали."""

    def __init__(self, replies=None):
        self.calls: list[list[dict[str, str]]] = []
        self._replies = list(replies or [])
        self.before_reply = None

    async def complete(self, messages, **kwargs):
        self.calls.append(messages)
        if self.before_reply is not None:
            self.before_reply()
        if self._replies:
            return self._replies.pop(0)
        return f"Ответ {len(self.calls)}."


@pytest.fixture()
def loop():
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


def build(replies=None, **cfg):
    """Очередь, её клиент и список озвученного."""
    client = FakeClient(replies)
    spoken: list[str] = []

    async def on_reply(task, reply):
        spoken.append(reply)

    return LlmQueue(FakeSection(**cfg), client, on_reply), client, spoken


def roles(messages) -> list[str]:
    return [m["role"] for m in messages]


async def ask(
    queue: LlmQueue, text: str, channel: int = CHANNEL, ended_at: float | None = None
) -> None:
    """Отправить реплику и дождаться, пока очередь отработает её целиком.

    ``ended_at`` — когда человек договорил. Явно его задают там, где важно, что
    реплика прозвучала уже после команды молчания: шаг monotonic() на Windows до
    16 мс, и в тесте две соседние строки попадают в один тик.
    """
    await queue.submit(
        ChatTask(
            ended_at=time.monotonic() if ended_at is None else ended_at,
            channel_id=channel,
            speaker="Вася",
            text=text,
        )
    )
    await queue._queue.join()
    # Озвучка идёт отдельной задачей — дадим ей доехать.
    await asyncio.gather(*list(queue._deliveries), return_exceptions=True)


def run(loop, queue, scenario) -> None:
    async def main():
        queue.start()
        try:
            await scenario()
        finally:
            await queue.stop()

    loop.run_until_complete(asyncio.wait_for(main(), 5.0))


# -------------------------------------------------------------- приоритет --
def test_who_finished_speaking_first_is_answered_first(loop):
    """Порядок задаёт конец фразы, а не момент попадания в очередь.

    Реплика, договорённая раньше, может доехать до очереди позже: у её автора
    сегмент длиннее, значит и STT считал дольше. Отвечать всё равно нужно ему
    первым — иначе тот, кто говорит короткими фразами, всегда влезает вперёд.
    """
    queue, client, _ = build()
    now = time.monotonic()

    async def scenario():
        # Кладём «поздний» первым: если приоритет сломан, он же первым и уйдёт.
        await queue.submit(
            ChatTask(ended_at=now, channel_id=CHANNEL, speaker="Петя", text="договорил позже")
        )
        await queue.submit(
            ChatTask(ended_at=now - 1.0, channel_id=CHANNEL, speaker="Вася", text="договорил раньше")
        )
        await queue._queue.join()
        await asyncio.gather(*list(queue._deliveries), return_exceptions=True)

    run(loop, queue, scenario)

    asked = [call[-1]["content"] for call in client.calls]
    assert asked == ["договорил раньше", "договорил позже"]


# ---------------------------------------------------------------- история --
def test_previous_exchange_reaches_the_model(loop):
    queue, client, spoken = build()

    async def scenario():
        await ask(queue, "я сломал стул")
        await ask(queue, "и что теперь")

    run(loop, queue, scenario)

    assert spoken == ["Ответ 1.", "Ответ 2."]
    # Второй запрос: системная часть, прошлый обмен, новый вопрос.
    assert roles(client.calls[1]) == ["system", "user", "assistant", "user"]
    assert client.calls[1][1]["content"] == "я сломал стул"
    assert client.calls[1][2]["content"] == "Ответ 1."


def test_speaker_name_stays_out_of_history(loop):
    """Ник не должен попадать в контекст: иначе баг возвращается окольным путём."""
    queue, client, _ = build()

    async def scenario():
        await ask(queue, "какая игра")
        await ask(queue, "а точнее")

    run(loop, queue, scenario)

    assert all("Вася" not in m["content"] for m in client.calls[1] if m["role"] != "system")


def test_history_keeps_only_the_last_turns(loop):
    queue, client, _ = build(history_turns=2)

    async def scenario():
        for text in ("первый", "второй", "третий", "четвёртый"):
            await ask(queue, text)

    run(loop, queue, scenario)

    context = " ".join(m["content"] for m in client.calls[-1])
    assert "первый" not in context, "старые ходы обязаны вытесняться"
    assert "второй" in context and "третий" in context


def test_history_is_dropped_after_ttl(loop):
    queue, client, _ = build(history_ttl_s=0.0)

    async def scenario():
        await ask(queue, "первый вопрос")
        await ask(queue, "второй вопрос")

    run(loop, queue, scenario)

    assert roles(client.calls[1]) == ["system", "user"]


def test_history_is_separate_per_channel(loop):
    queue, client, _ = build()

    async def scenario():
        await ask(queue, "вопрос в первом канале", channel=1)
        await ask(queue, "вопрос во втором", channel=2)

    run(loop, queue, scenario)

    assert roles(client.calls[1]) == ["system", "user"]


def test_empty_reply_does_not_break_alternation(loop):
    """Пустой ответ в историю не пишется: иначе поедет чередование ролей."""
    queue, client, spoken = build(replies=["   ", "Норм."])

    async def scenario():
        await ask(queue, "молчаливый вопрос")
        await ask(queue, "следующий вопрос")

    run(loop, queue, scenario)

    assert spoken == ["Норм."]
    assert roles(client.calls[1]) == ["system", "user"]


def test_history_stores_the_trimmed_reply(loop):
    """В контекст идёт то, что реально прозвучало, а не всё, что выдала модель."""
    queue, client, spoken = build(replies=["Первое. Второе.", "Ага."])

    async def scenario():
        await ask(queue, "вопрос")
        await ask(queue, "ещё вопрос")

    run(loop, queue, scenario)

    assert spoken == ["Первое.", "Ага."]
    assert client.calls[1][2]["content"] == "Первое."


# ------------------------------------------------------------------ drop --
def test_drop_discards_queued_replies(loop):
    """Команда молчания: то, что стоит в очереди, разбирать уже незачем."""
    queue, client, spoken = build()

    async def scenario():
        await queue.submit(
            ChatTask(ended_at=time.monotonic(), channel_id=CHANNEL, speaker="Вася", text="вопрос")
        )
        queue.drop(CHANNEL)
        await queue._queue.join()

    run(loop, queue, scenario)

    assert spoken == []
    assert client.calls == [], "выброшенная реплика не должна доходить до модели"


def test_drop_discards_a_reply_that_was_already_computed(loop):
    """Команда молчания пришла, пока модель считала: ответ выбрасывается."""
    queue, client, spoken = build()
    client.before_reply = lambda: queue.drop(CHANNEL)

    async def scenario():
        await ask(queue, "долгий вопрос")

    run(loop, queue, scenario)

    assert spoken == []


def test_question_after_the_shut_up_is_answered_from_scratch(loop):
    """После команды молчания следующий вопрос обслуживается без контекста."""
    queue, client, spoken = build()

    async def scenario():
        await ask(queue, "первый вопрос")
        queue.drop(CHANNEL)
        await ask(queue, "второй вопрос", ended_at=time.monotonic() + 1.0)

    run(loop, queue, scenario)

    assert spoken == ["Ответ 1.", "Ответ 2."]
    assert roles(client.calls[1]) == ["system", "user"]


def test_drop_touches_only_its_own_channel(loop):
    queue, client, _ = build()

    async def scenario():
        await ask(queue, "вопрос в первом", channel=1)
        queue.drop(2)
        await ask(queue, "второй вопрос в первом", channel=1)

    run(loop, queue, scenario)

    assert roles(client.calls[1]) == ["system", "user", "assistant", "user"]
