"""Поиск имени «Башмак» в расшифровке.

Отдельную wake-word модель не тренируем: STT и так расшифровывает всю речь,
которую выделил VAD, поэтому достаточно нечёткого поиска по тексту. Whisper
регулярно слышит «башмака», «бошмак», «башмачок» — точное сравнение тут
бесполезно, а rapidfuzz ловит все склонения и типичные ошибки распознавания.

Сравнение пословное, а не по всей строке: ``fuzz.ratio`` по длинной фразе
не сработал бы вовсе, а ``partial_ratio`` дал бы ложные срабатывания на любом
похожем куске — нам нужно именно обращение.

Нормализация — только для сравнения. Наружу уходит исходный текст с вырезанным
именем: LLM и поиск музыки должны получить «Расскажи анекдот про кота!», а не
«расскажи анекдот про кота».
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass

from rapidfuzz import fuzz

log = logging.getLogger(__name__)

_PUNCT = re.compile(r"[^\w\s-]+", re.UNICODE)
_SPACES = re.compile(r"\s+")
#: Слово в исходном тексте — по нему ищем имя, не теряя позиций в строке.
_WORD = re.compile(r"[\w-]+", re.UNICODE)
#: Что срезать с краёв запроса после вырезания имени.
_EDGE = " \t\n\r,.!?:;-—–…\"'«»"


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
        if not text or not text.strip():
            return None

        # Ищем по исходной строке, сравниваем по нормализованным словам: так
        # известны позиции, и запрос можно вырезать как есть, без потери
        # регистра и пунктуации.
        best: re.Match[str] | None = None
        best_score = 0.0

        for found in _WORD.finditer(text):
            word = normalize(found.group(0))
            # Длина отсекает мусор до вызова fuzz: «а», «ну», «да» имени не ровня.
            if len(word) < 4:
                continue
            score = max(fuzz.ratio(word, variant) for variant in self.variants)
            if score > best_score:
                best_score, best = score, found

        if best is None or best_score < self.threshold:
            return None

        if self.strip_prefix:
            # Слева срезаем пунктуацию («Башмак, привет» → «привет»), справа
            # только пробелы: «!» и «?» в конце запроса — часть фразы.
            payload = text[best.end() :].lstrip(_EDGE).rstrip()
            # «Башмак!» без продолжения — тоже обращение, пусть отзовётся.
            if not payload:
                payload = text[: best.start()].strip(_EDGE)
        else:
            payload = text.strip()

        return WakeMatch(payload=_SPACES.sub(" ", payload), score=best_score, word=best.group(0))
