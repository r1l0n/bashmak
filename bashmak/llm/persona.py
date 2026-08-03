"""Персона «Башмака» и сборка промпта.

Главное ограничение, о котором легко забыть: ответ идёт не в чат, а в
синтезатор речи. Списки, markdown, ссылки, смайлики и длинные абзацы Piper
озвучит буквально — получится мусор. Поэтому запреты на форматирование в
системном промпте не косметика, а требование пайплайна.
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass

SYSTEM_PROMPT = """Ты — Башмак, голосовой помощник в голосовом канале Discord.

Как говорить:
- Коротко. Одно-три предложения, если не просят подробнее.
- Живой разговорной речью, на «ты», с юмором, но без хамства.
- Только русским языком, если с тобой не заговорили на другом.

Формат ответа (это важно — тебя озвучивает синтезатор речи):
- Обычный текст без разметки. Никаких списков, звёздочек, решёток, ссылок и смайликов.
- Числа и сокращения пиши словами, если так их проще произнести.
- Не описывай действия и не ставь ремарки в скобках.

В канале несколько человек, перед репликой указано имя говорящего.
Обращайся по имени, когда это уместно. Имя в начале чужой реплики — это
пометка, а не часть текста."""


@dataclass(slots=True)
class Turn:
    role: str  # "user" | "assistant"
    speaker: str  # ник говорящего, для ассистента — пусто
    text: str
    at: float


#: Сколько подслушанных реплик держать как фон. Больше не нужно: это чужой
#: разговор, а не диалог с ботом.
AMBIENT_LINES = 4


class Conversation:
    """История одного голосового канала.

    Скользящее окно, а не полный диалог: контекст 4096 токенов при 7B на CPU
    — это ещё и время на prefill, которое пользователь ждёт молча.

    Двумя окнами, а не одним. Реплики, адресованные боту, и его ответы — это
    диалог, его нужно помнить. Всё остальное, что звучало в канале (включая
    мусор от Whisper), — фон: полезен как контекст текущего вопроса, но
    вытеснять им диалог нельзя, иначе трое болтающих людей за пару секунд
    сотрут и вопрос, и собственный ответ бота.
    """

    def __init__(self, max_turns: int = 12, ambient_lines: int = AMBIENT_LINES) -> None:
        self._turns: deque[Turn] = deque(maxlen=max_turns)
        self._ambient: deque[Turn] = deque(maxlen=max(0, ambient_lines))

    def add_user(self, speaker: str, text: str) -> None:
        self._turns.append(Turn("user", speaker, text, time.time()))
        # Фон уже уехал в промпт вместе с этой репликой — второй раз не нужен.
        self._ambient.clear()

    def add_assistant(self, text: str) -> None:
        self._turns.append(Turn("assistant", "", text, time.time()))

    def note_ambient(self, speaker: str, text: str) -> None:
        """Реплика не боту: запомнить как фон разговора."""
        if self._ambient.maxlen:
            self._ambient.append(Turn("user", speaker, text, time.time()))

    def clear(self) -> None:
        self._turns.clear()
        self._ambient.clear()

    def build_messages(self, speaker: str, text: str) -> list[dict[str, str]]:
        """Системный промпт + история + фон + текущая реплика."""
        messages: list[dict[str, str]] = [{"role": "system", "content": SYSTEM_PROMPT}]

        def push(role: str, content: str) -> None:
            # Подряд идущие реплики одной роли склеиваем: серия отдельных
            # user-сообщений сбивает шаблон чата у instruct-моделей.
            if messages and messages[-1]["role"] == role and role != "system":
                messages[-1]["content"] += "\n" + content
            else:
                messages.append({"role": role, "content": content})

        for turn in self._turns:
            push(turn.role, f"{turn.speaker}: {turn.text}" if turn.role == "user" else turn.text)
        for turn in self._ambient:
            push("user", f"{turn.speaker}: {turn.text}")
        push("user", f"{speaker}: {text}")
        return messages
