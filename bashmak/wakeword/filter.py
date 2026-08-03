"""Поиск имени «Башмак» в расшифровке.

Отдельную wake-word модель не тренируем: STT и так расшифровывает всю речь,
которую выделил VAD, поэтому достаточно нечёткого поиска по тексту. Whisper
регулярно слышит «башмака», «бошмак», «башмачок» — точное сравнение тут
бесполезно, а rapidfuzz ловит все склонения и типичные ошибки распознавания.

Сравнение пословное, а не по всей строке: ``partial_ratio`` по длинной фразе
даёт ложные срабатывания на любом похожем куске, а нам нужно именно
обращение.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass

from rapidfuzz import fuzz

log = logging.getLogger(__name__)

_PUNCT = re.compile(r"[^\w\s-]+", re.UNICODE)
_SPACES = re.compile(r"\s+")


def normalize(text: str) -> str:
    """Нижний регистр, ё→е, без пунктуации — в таком виде и сравниваем."""
    lowered = text.lower().replace("ё", "е")
    return _SPACES.sub(" ", _PUNCT.sub(" ", lowered)).strip()


@dataclass(slots=True)
class WakeMatch:
    """Что осталось от фразы после обращения."""

    payload: str  # текст запроса к боту
    score: float  # насколько уверенно опознано имя, 0..100
    word: str  # как имя прозвучало на самом деле


class WakeWordFilter:
    def __init__(self, cfg) -> None:  # noqa: ANN001 — bashmak.config.Section
        self.variants = [normalize(v) for v in cfg.get("variants", ["башмак"])]
        self.threshold = float(cfg.get("threshold", 80))
        self.strip_prefix = bool(cfg.get("strip_prefix", True))

    def match(self, text: str) -> WakeMatch | None:
        """None — обращения нет, реплику дальше по пайплайну не пускаем."""
        normalized = normalize(text)
        if not normalized:
            return None

        words = normalized.split(" ")
        best_index = -1
        best_score = 0.0

        for index, word in enumerate(words):
            # Длина отсекает мусор до вызова fuzz: «а», «ну», «да» имени не ровня.
            if len(word) < 4:
                continue
            score = max(fuzz.ratio(word, variant) for variant in self.variants)
            if score > best_score:
                best_score, best_index = score, index

        if best_index < 0 or best_score < self.threshold:
            return None

        if self.strip_prefix:
            payload = " ".join(words[best_index + 1 :]).strip()
            # «Башмак!» без продолжения — тоже обращение, пусть отзовётся.
            if not payload:
                payload = " ".join(words[:best_index]).strip()
        else:
            payload = normalized

        return WakeMatch(payload=payload, score=best_score, word=words[best_index])
