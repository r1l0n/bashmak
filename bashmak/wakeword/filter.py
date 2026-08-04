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

Имя может стоять где угодно: «Башмак, включи дору», «включи дору, Башмак»,
«включи дору, Башмак, пожалуйста». Поэтому вырезаются все его вхождения, а
остаток склеивается обратно, — а не берётся «всё после имени».
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
        hits: list[tuple[re.Match[str], float]] = []

        for found in _WORD.finditer(text):
            word = normalize(found.group(0))
            # Длина отсекает мусор до вызова fuzz: «а», «ну», «да» имени не ровня.
            if len(word) < 4:
                continue
            score = max(fuzz.ratio(word, variant) for variant in self.variants)
            if score >= self.threshold:
                hits.append((found, score))

        if not hits:
            return None

        # Наружу отдаём лучшее совпадение: на нём висит отладка «как имя
        # прозвучало на самом деле».
        best, best_score = max(hits, key=lambda hit: hit[1])

        payload = self._strip_name(text, hits) if self.strip_prefix else text.strip()
        return WakeMatch(payload=_SPACES.sub(" ", payload), score=best_score, word=best.group(0))

    @staticmethod
    def _strip_name(text: str, hits: list[tuple[re.Match[str], float]]) -> str:
        """Вырезать все вхождения имени и склеить остаток фразы.

        Имя — обращение, а не начало команды: оно бывает и в конце («включи
        дору, Башмак»), и в середине («включи дору, Башмак, пожалуйста»), и
        дважды за фразу. Поэтому берём не «всё после имени», а всё, кроме
        самого имени, — иначе половина запроса молча теряется.
        """
        pieces: list[str] = []
        cursor = 0
        for found, _ in hits:
            pieces.append(text[cursor : found.start()])
            cursor = found.end()
        pieces.append(text[cursor:])

        # У всех кусков, кроме последнего, срезаем пунктуацию с обеих сторон:
        # иначе на стыке вырезанного имени получится «дору, , пожалуйста».
        parts = [piece.strip(_EDGE) for piece in pieces[:-1]]
        # А у последнего правый край не трогаем: «!» и «?» в конце — часть
        # фразы, и модель по ним слышит интонацию.
        parts.append(pieces[-1].lstrip(_EDGE).rstrip())
        return ", ".join(part for part in parts if part)
