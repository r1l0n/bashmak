"""Очередь запросов к LLM.

На CPU параллельный инференс одной модели не ускоряет, а замедляет: потоки
дерутся за те же ядра и кеш. Поэтому llama-server один, и обращение к нему
строго по одному — очередь и есть механизм throttling'а.

Очередь приоритетная по моменту окончания фразы: кто первым договорил, тому
первым и отвечаем, даже если его STT отработал дольше соседского. Реплики,
пролежавшие дольше ``stale_task_s``, выбрасываются — отвечать через минуту
на «что там по погоде» уже незачем.

История короткая и общая на голосовой канал: люди в нём разговаривают с одним
Башмаком и подхватывают чужие вопросы. Держим её здесь, а не в persona.py:
сборка промпта остаётся без состояния. Глубина маленькая намеренно — на CPU
длинный контекст разбавляет вопрос, и модель начинает отвечать на фон.

``drop(channel_id)`` — «завали ебало»: всё, что ещё не прозвучало в канале,
выбрасывается вместе с историей.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Awaitable, Callable

from ..utils.logging import current_cid, guard, stage, turn_note
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
        #: Сколько обменов (вопрос + ответ) помнить в канале.
        self._history_turns = int(cfg.get("history_turns", 3))
        self._history_ttl = float(cfg.get("history_ttl_s", 120))

        self._queue: asyncio.PriorityQueue[ChatTask] = asyncio.PriorityQueue()
        self._task: asyncio.Task | None = None
        #: Задача озвучки → канал, чтобы drop() гасил только свой канал.
        self._deliveries: dict[asyncio.Task, int] = {}
        self._history: dict[int, deque[dict[str, str]]] = {}
        self._history_at: dict[int, float] = {}
        #: Канал → момент «замолчи». Всё, что договорили до него, не отвечаем.
        self._muted: dict[int, float] = {}

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

    def drop(self, channel_id: int) -> None:
        """Забыть всё, что ещё не прозвучало в канале, и обнулить контекст.

        Отсечка по времени, а не только отмена задач: реплика может быть прямо
        сейчас в инференсе, и оборвать HTTP-запрос дешевле не выйдет — зато
        готовый ответ на неё уже никому не нужен.
        """
        self._muted[channel_id] = time.monotonic()
        for delivery, channel in list(self._deliveries.items()):
            if channel == channel_id:
                delivery.cancel()
        self._history.pop(channel_id, None)
        self._history_at.pop(channel_id, None)

    # ---------------------------------------------------------- история ---
    def _history_for(self, channel_id: int) -> deque[dict[str, str]]:
        """Контекст канала. Протухший по времени — пустой."""
        history = self._history.get(channel_id)
        if history is None:
            # Пары user+assistant, поэтому длина вдвое больше числа обменов.
            history = deque(maxlen=max(0, self._history_turns) * 2)
            self._history[channel_id] = history
        last = self._history_at.get(channel_id)
        # Нестрого, чтобы history_ttl_s: 0 означало «без истории вовсе».
        if last is not None and time.monotonic() - last >= self._history_ttl:
            log.debug("контекст канала %s протух, начинаю разговор заново", channel_id)
            history.clear()
        return history

    def _muted_for(self, task: ChatTask) -> bool:
        """Успели сказать «заткнись» после того, как человек договорил?

        Сравнение нестрогое: шаг monotonic() кое-где доходит до 16 мс, и
        «заткнись» запросто попадает в тот же тик, что и реплика, которую оно
        отменяет. Новую реплику в этот зазор не втиснуть — до неё минимум
        min_speech_ms речи, silence_ms тишины и целое распознавание.
        """
        muted_at = self._muted.get(task.channel_id)
        return muted_at is not None and muted_at >= task.ended_at

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
                if self._muted_for(task):
                    log.info("велели молчать — реплика из очереди выброшена: %r", task.text)
                    continue
                await self._handle(task)
            except Exception:
                log.exception("не смог обработать реплику: %r", task.text)
            finally:
                self._queue.task_done()

    async def _handle(self, task: ChatTask) -> None:
        # До запроса, а не после: если модель упадёт, в отчёте всё равно должно
        # быть видно, что именно ей отправляли. Без имени говорящего — оно
        # уходит в системную часть, а в шапке отчёта и так стоит.
        turn_note(sent=task.text)

        history = self._history_for(task.channel_id)
        with stage(log, "llm", queue=self._queue.qsize(), контекст=len(history)) as info:
            raw = await self._client.complete(
                build_messages(task.speaker, task.text, history)
            )
            info["chars"] = len(raw)

        if self._muted_for(task):
            # Велели молчать, пока модель думала. Ответ не озвучиваем и в
            # историю не пишем — разговора, к которому он относится, уже нет.
            log.info("велели молчать — готовый ответ выброшен: %r", raw)
            return

        reply = clean_reply(raw)
        if reply != raw:
            # Модель либо пошла дописывать диалог за собеседника (и тогда в
            # канал уехала бы выдуманная беседа целиком), либо не удержалась в
            # одном предложении. Второе — рутина, шуметь про него незачем.
            log.debug("ответ обрезан: %r", raw)
        if not reply:
            log.warning("LLM вернула пустой ответ на %r", task.text)
            return

        turn_note(reply=reply)

        # Обе реплики разом и только после удачного ответа: пустой или
        # оборванный ход сломал бы чередование user/assistant, которое
        # проверяет LlmClient._check_roles. В историю кладём то, что реально
        # прозвучит, — то есть уже обрезанный ответ.
        history.append({"role": "user", "content": task.text})
        history.append({"role": "assistant", "content": reply})
        self._history_at[task.channel_id] = time.monotonic()

        # Озвучка идёт отдельным таском: ждать её здесь значило бы держать
        # очередь простаивающей всё время проигрывания (ответ на 15 секунд —
        # 15 секунд простоя), а stale_task_s тем временем выбрасывал бы уже
        # распознанные реплики. Порядок реплик держит лок арбитра.
        delivery = asyncio.create_task(self._deliver(task, reply), name="llm-reply")
        self._deliveries[delivery] = task.channel_id
        delivery.add_done_callback(lambda done: self._deliveries.pop(done, None))

    async def _deliver(self, task: ChatTask, reply: str) -> None:
        # Ещё одна проверка: «заткнись» могло прилететь, пока задача ждала
        # своей очереди на озвучку за чужой репликой (лок арбитра).
        if self._muted_for(task):
            log.debug("велели молчать — ответ до озвучки не доехал: %r", reply)
            return
        try:
            await self._on_reply(task, reply)
        except asyncio.CancelledError:
            raise
        except Exception:
            log.exception("не смог доставить ответ: %r", reply)
