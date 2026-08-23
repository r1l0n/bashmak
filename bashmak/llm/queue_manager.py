"""Очередь запросов к LLM.

На CPU параллельный инференс одной модели замедляет работу: потоки конкурируют
за те же ядра и кеш. Поэтому llama-server один и обращение к нему строго
последовательное — очередь работает механизмом throttling'а.

Приоритет — по моменту окончания фразы: кто первым договорил, тому первым и
отвечаем, даже если его STT отработал дольше. Реплики, пролежавшие дольше
``stale_task_s``, выбрасываются как неактуальные.

История общая на голосовой канал: участники разговаривают с одним ботом и
подхватывают чужие вопросы. Хранится здесь, а не в persona.py, чтобы сборка
промпта осталась без состояния. Глубина ограничена: на CPU длинный контекст
разбавляет вопрос, и модель начинает отвечать на фон.

``drop(channel_id)`` — обработка команды молчания: всё, что ещё не прозвучало
в канале, выбрасывается вместе с историей.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Awaitable, Callable

from ..utils.logging import QUEUE_STAGE, current_cid, guard, stage, turn_note, turn_stage
from .client import CHAT_SLOT, LlmClient
from .persona import build_messages, clean_reply

log = logging.getLogger(__name__)


def _repeats(history: deque[dict[str, str]], reply: str) -> bool:
    """Говорил ли бот это же в последних ходах канала.

    Сравнение по всей истории, а не только с прошлым ответом: зацикливается
    модель и через ход («А», «Б», «А»). Регистр и знаки в конце не считаются —
    повтор от повтора отличается ровно ими.
    """
    same = reply.casefold().strip(" .!?…")
    return any(
        message["role"] == "assistant" and message["content"].casefold().strip(" .!?…") == same
        for message in history
    )


@dataclass(order=True)
class ChatTask:
    """Заявка на ответ. Сравнивается только по ended_at — это и есть FIFO."""

    ended_at: float
    channel_id: int = field(compare=False, default=0)
    user_id: int = field(compare=False, default=0)
    speaker: str = field(compare=False, default="")
    text: str = field(compare=False, default="")
    cid: str = field(compare=False, default="-")
    #: Момент постановки в очередь; проставляет submit(). Отдельно от ended_at:
    #: тот задаёт приоритет, а этот нужен, чтобы отделить ожидание за чужим
    #: инференсом от времени самого инференса. compare=False обязателен —
    #: датакласс сравнивается только по ended_at, это и есть порядок ответов.
    queued_at: float = field(compare=False, default=0.0)


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
        task.queued_at = time.monotonic()
        await self._queue.put(task)
        log.debug("в очередь LLM: %r (глубина %d)", task.text, self._queue.qsize())

    @property
    def depth(self) -> int:
        return self._queue.qsize()

    def drop(self, channel_id: int) -> None:
        """Забыть всё, что ещё не прозвучало в канале, и обнулить контекст.

        Отсечка по времени, а не только отмена задач: реплика может быть прямо
        сейчас в инференсе, оборвать HTTP-запрос дешевле не выйдет, но готовый
        ответ на неё уже не нужен.
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
        """Прозвучала ли команда молчания после того, как участник договорил.

        Сравнение нестрогое: шаг monotonic() местами доходит до 16 мс, и
        команда попадает в тот же тик, что и реплика, которую она отменяет.
        Новая реплика в этот зазор не помещается: до неё минимум min_speech_ms
        речи, silence_ms тишины и целое распознавание.
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
                    log.info("команда молчания — реплика из очереди выброшена: %r", task.text)
                    continue
                await self._handle(task)
            except Exception:
                log.exception("не смог обработать реплику: %r", task.text)
            finally:
                self._queue.task_done()

    async def _handle(self, task: ChatTask) -> None:
        # Сколько реплика простояла за чужим инференсом. Отдельной стадией,
        # иначе это время растворяется в «всего» и выглядит как медленная LLM.
        if task.queued_at:
            turn_stage(QUEUE_STAGE, time.monotonic() - task.queued_at)

        # До запроса, а не после: если модель упадёт, в отчёте всё равно должно
        # быть видно, что именно ей отправляли. Без имени говорящего — оно
        # уходит в системную часть, а в шапке отчёта и так стоит.
        turn_note(sent=task.text)

        history = self._history_for(task.channel_id)
        with stage(log, "llm", queue=self._queue.qsize(), контекст=len(history)) as info:
            raw = await self._client.complete(
                build_messages(task.speaker, task.text, history),
                # Свой слот: классификатор намерений не должен вытирать кеш
                # префикса диалога (см. CHAT_SLOT).
                slot=CHAT_SLOT,
            )
            info["chars"] = len(raw)

        if self._muted_for(task):
            # Команда молчания пришла, пока модель считала. Ответ не озвучиваем
            # и в историю не пишем: разговора, к которому он относится, уже нет.
            log.info("команда молчания — готовый ответ выброшен: %r", raw)
            return

        reply = clean_reply(raw)
        if reply != raw:
            # Модель дописала диалог за собеседника или налепила разметки — без
            # чистки в канал ушла бы выдуманная беседа целиком.
            log.debug("ответ обрезан: %r", raw)
        if not reply:
            log.warning("LLM вернула пустой ответ на %r", task.text)
            return

        turn_note(reply=reply)

        if _repeats(history, reply):
            # Модель зациклилась на собственном ответе. Само оно не проходит:
            # повтор лежит в истории и на следующем ходу подсказывает себя же,
            # так что каждый следующий ответ только вероятнее. Раньше это
            # разруливалось вручную — «Башмак, заткнись» чистит контекст через
            # drop(); здесь то же самое, но само и на первом же повторе.
            log.info("модель повторяется — сбрасываю контекст канала %s", task.channel_id)
            history.clear()
            self._history_at.pop(task.channel_id, None)
        else:
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
        # Ещё одна проверка: команда молчания могла прийти, пока задача ждала
        # своей очереди на озвучку за чужой репликой (лок арбитра).
        if self._muted_for(task):
            log.debug("команда молчания — ответ до озвучки не дошёл: %r", reply)
            return
        try:
            await self._on_reply(task, reply)
        except asyncio.CancelledError:
            raise
        except Exception:
            log.exception("не смог доставить ответ: %r", reply)
