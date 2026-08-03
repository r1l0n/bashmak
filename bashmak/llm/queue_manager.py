"""Очередь запросов к LLM.

На CPU параллельный инференс одной модели не ускоряет, а замедляет: потоки
дерутся за те же ядра и кеш. Поэтому llama-server один, и обращение к нему
строго по одному — очередь и есть механизм throttling'а.

Очередь приоритетная по моменту окончания фразы: кто первым договорил, тому
первым и отвечаем, даже если его STT отработал дольше соседского. Реплики,
пролежавшие дольше ``stale_task_s``, выбрасываются — отвечать через минуту
на «что там по погоде» уже незачем.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Awaitable, Callable

from ..utils.logging import current_cid, guard, stage
from .client import LlmClient
from .persona import Conversation

log = logging.getLogger(__name__)

#: Сколько каналов держать в памяти. История чистится при выходе из канала,
#: но лимит страхует от утечки, если сессия закрылась не по-хорошему.
MAX_CONVERSATIONS = 32


@dataclass(order=True)
class ChatTask:
    """Заявка на ответ. Сравнивается только по ended_at — это и есть FIFO."""

    ended_at: float
    channel_id: int = field(compare=False, default=0)
    user_id: int = field(compare=False, default=0)
    speaker: str = field(compare=False, default="")
    text: str = field(compare=False, default="")
    cid: str = field(compare=False, default="-")


class LlmQueue:
    def __init__(
        self,
        cfg,  # noqa: ANN001 — bashmak.config.Section (llm)
        client: LlmClient,
        on_reply: Callable[[ChatTask, str], Awaitable[None]],
    ) -> None:
        self._client = client
        self._on_reply = on_reply
        self._stale_after = float(cfg.get("stale_task_s", 45))
        self._history_turns = int(cfg.get("history_turns", 12))

        self._queue: asyncio.PriorityQueue[ChatTask] = asyncio.PriorityQueue()
        self._conversations: OrderedDict[int, Conversation] = OrderedDict()
        self._task: asyncio.Task | None = None
        self._deliveries: set[asyncio.Task] = set()

    # ------------------------------------------------------------- жизнь --
    def start(self) -> None:
        if self._task is None or self._task.done():
            self._task = asyncio.create_task(self._run(), name="llm-queue")
            log.info("очередь LLM запущена")

    async def stop(self) -> None:
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None

        pending = list(self._deliveries)
        for delivery in pending:
            delivery.cancel()
        if pending:
            await asyncio.gather(*pending, return_exceptions=True)
        self._deliveries.clear()
        log.info("очередь LLM остановлена")

    # ------------------------------------------------------------- API ----
    async def submit(self, task: ChatTask) -> None:
        await self._queue.put(task)
        log.debug("в очередь LLM: %r (глубина %d)", task.text, self._queue.qsize())

    def conversation(self, channel_id: int) -> Conversation:
        conversation = self._conversations.get(channel_id)
        if conversation is None:
            conversation = Conversation(self._history_turns)
            self._conversations[channel_id] = conversation
            while len(self._conversations) > MAX_CONVERSATIONS:
                dropped, _ = self._conversations.popitem(last=False)
                log.debug("история канала %s вытеснена по лимиту", dropped)
        self._conversations.move_to_end(channel_id)
        return conversation

    def note_user_line(self, channel_id: int, speaker: str, text: str) -> None:
        """Запомнить реплику без обращения к боту — как фон разговора.

        Именно фон, а не полноценный ход диалога: иначе трое болтающих людей
        за пару секунд вытеснили бы из окна истории и вопрос, и свой же ответ.
        """
        self.conversation(channel_id).note_ambient(speaker, text)

    def reset(self, channel_id: int) -> None:
        self._conversations.pop(channel_id, None)

    @property
    def depth(self) -> int:
        return self._queue.qsize()

    # ------------------------------------------------------------ работа --
    @guard("очередь LLM", reraise=False)
    async def _run(self) -> None:
        while True:
            task = await self._queue.get()
            current_cid.set(task.cid)
            try:
                age = time.monotonic() - task.ended_at
                if age > self._stale_after:
                    log.warning("реплика протухла (%.0f с в очереди), выброшена: %r", age, task.text)
                    continue
                await self._handle(task)
            except Exception:
                log.exception("не смог обработать реплику: %r", task.text)
            finally:
                self._queue.task_done()

    async def _handle(self, task: ChatTask) -> None:
        conversation = self.conversation(task.channel_id)
        messages = conversation.build_messages(task.speaker, task.text)

        with stage(log, "llm", queue=self._queue.qsize()) as info:
            reply = await self._client.complete(messages)
            info["chars"] = len(reply)

        if not reply:
            log.warning("LLM вернула пустой ответ на %r", task.text)
            return

        conversation.add_user(task.speaker, task.text)
        conversation.add_assistant(reply)
        log.info("ответ: %r", reply)

        # Озвучка идёт отдельным таском: ждать её здесь значило бы держать
        # очередь простаивающей всё время проигрывания (ответ на 15 секунд —
        # 15 секунд простоя), а stale_task_s тем временем выбрасывал бы уже
        # распознанные реплики. Порядок реплик держит лок арбитра.
        delivery = asyncio.create_task(self._deliver(task, reply), name="llm-reply")
        self._deliveries.add(delivery)
        delivery.add_done_callback(self._deliveries.discard)

    async def _deliver(self, task: ChatTask, reply: str) -> None:
        try:
            await self._on_reply(task, reply)
        except asyncio.CancelledError:
            raise
        except Exception:
            log.exception("не смог доставить ответ: %r", reply)
