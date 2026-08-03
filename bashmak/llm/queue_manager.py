"""Очередь запросов к LLM.

На CPU параллельный инференс одной модели не ускоряет, а замедляет: потоки
дерутся за те же ядра и кеш. Поэтому llama-server один, и обращение к нему
строго по одному — очередь и есть механизм throttling'а.

Очередь приоритетная по моменту окончания фразы: кто первым договорил, тому
первым и отвечаем, даже если его STT отработал дольше соседского. Реплики,
пролежавшие дольше ``stale_task_s``, выбрасываются — отвечать через минуту
на «что там по погоде» уже незачем.

Истории между запросами нет: каждая реплика уходит в модель одна (см.
persona.py). Хранить тут нечего, поэтому очередь без состояния.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Awaitable, Callable

from ..utils.logging import current_cid, guard, stage
from .client import LlmClient
from .persona import build_messages, clean_reply

log = logging.getLogger(__name__)


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

        self._queue: asyncio.PriorityQueue[ChatTask] = asyncio.PriorityQueue()
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
        with stage(log, "llm", queue=self._queue.qsize()) as info:
            raw = await self._client.complete(build_messages(task.speaker, task.text))
            info["chars"] = len(raw)

        reply = clean_reply(raw)
        if reply != raw:
            # Не мелочь: значит, модель пошла дописывать диалог за собеседника,
            # и в голосовой канал уехала бы выдуманная беседа целиком.
            log.warning("в ответе была чужая разметка, обрезал: %r", raw)
        if not reply:
            log.warning("LLM вернула пустой ответ на %r", task.text)
            return

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
