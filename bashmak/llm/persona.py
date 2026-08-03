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


class Conversation:
    """История одного голосового канала.

    Скользящее окно, а не полный диалог: контекст 4096 токенов при 7B на CPU
    — это ещё и время на prefill, которое пользователь ждёт молча.
    """

    def __init__(self, max_turns: int = 12) -> None:
        self._turns: deque[Turn] = deque(maxlen=max_turns)

    def add_user(self, speaker: str, text: str) -> None:
        self._turns.append(Turn("user", speaker, text, time.time()))

    def add_assistant(self, text: str) -> None:
        self._turns.append(Turn("assistant", "", text, time.time()))

    def clear(self) -> None:
        self._turns.clear()

    def build_messages(self, speaker: str, text: str) -> list[dict[str, str]]:
        """Системный промпт + история + текущая реплика."""
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        for turn in self._turns:
            content = f"{turn.speaker}: {turn.text}" if turn.role == "user" else turn.text
            messages.append({"role": turn.role, "content": content})
        messages.append({"role": "user", "content": f"{speaker}: {text}"})
        return messages
