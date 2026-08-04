"""Что человек хотел: поболтать или порулить музыкой.

Два уровня:

1. Регекс по ключевым словам — мгновенно, без LLM. Ловит подавляющее
   большинство команд («поставь Кино», «следующий трек», «сделай потише»).
2. Классификация той же LLM — для формулировок, которые правила не разобрали.

Второй уровень по умолчанию включается не всегда, а только когда в реплике
есть хоть какой-то музыкальный намёк (``llm_fallback: hinted``). Иначе каждая
обычная болтовня оплачивала бы лишний инференс на CPU — то есть плюс секунду
к ответу на ровном месте.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from enum import Enum

from ..utils.logging import stage

log = logging.getLogger(__name__)


class Intent(str, Enum):
    CHAT = "chat"
    #: «Завали ебало» — замолчать и вырубить музыку, вслух не отвечая.
    SILENCE = "silence"
    MUSIC_PLAY = "music_play"
    MUSIC_PAUSE = "music_pause"
    MUSIC_RESUME = "music_resume"
    MUSIC_SKIP = "music_skip"
    MUSIC_STOP = "music_stop"
    MUSIC_LOUDER = "music_louder"
    MUSIC_QUIETER = "music_quieter"


@dataclass(slots=True)
class Decision:
    intent: Intent
    query: str = ""
    source: str = "regex"


# Порядок важен: «выключи музыку» должно поймать stop, а не play по «включ».
#
# Про подбор слов для SILENCE. «Тише»/«потише» не берём — это громкость
# («сделай потише»). Голое «хватит» не берём — оно занято правилом ниже
# («хватит музыки»). А «заглохни» и «тишина» переехали сюда из music_stop: по
# смыслу это «замолчи», а не «сними трек», — и музыку silence всё равно снимает.
_RULES: tuple[tuple[Intent, re.Pattern[str]], ...] = (
    (
        Intent.SILENCE,
        re.compile(
            # Только повелительные формы: «завалил экзамен» и «молчание» —
            # обычная речь, а не команда.
            r"\b(завали|заткн\w*|захлопни|заглохни|тишина|уймись|замолкни"
            r"|(?:за|по)?молчи|молчать|стоп"
            r"|не пизди|хорош пиздеть|хватит (?:болтать|говорить|трещать))\b"
        ),
    ),
    (Intent.MUSIC_STOP, re.compile(r"\b(выключ\w*|останов\w*|стоп|прекрат\w*|хватит)\b.*\b(музык\w*|трек\w*|песн\w*|плейлист\w*)")),
    (Intent.MUSIC_PAUSE, re.compile(r"\b(пауз\w*|приостанов\w*|погоди с музык\w*)\b")),
    (Intent.MUSIC_RESUME, re.compile(r"\b(продолж\w*|возобнов\w*|снова играй|с паузы)\b")),
    (Intent.MUSIC_SKIP, re.compile(r"\b(следующ\w*|дальше|скип\w*|пропусти|переключ\w*|другую песн\w*|другой трек)\b")),
    (Intent.MUSIC_LOUDER, re.compile(r"\b(по)?громче\b|\bприбав\w* (звук|громкост\w*)\b")),
    (Intent.MUSIC_QUIETER, re.compile(r"\b(по)?тише\b|\bубав\w* (звук|громкост\w*)\b")),
    (Intent.MUSIC_PLAY, re.compile(r"\b(включ\w*|постав\w*|запусти|врубн?и\w*|сыграй|играй|поищи|найди)\b")),
)

#: Явное упоминание музыки — по нему отличаем «включи музыку» от «включи мозг».
_MUSIC_NOUN = re.compile(r"\b(музык\w*|трек\w*|песн\w*|плейлист\w*|альбом\w*|композици\w*)\b")

#: Намерения, слова которых сплошь и рядом встречаются в обычной речи:
#: «Башмак, продолжай» — это не resume, «а что дальше?» — не skip, «говори
#: потише» — не громкость. Такие команды принимаем, только если музыка названа
#: явно или уже играет.
_NEEDS_MUSIC_CONTEXT = frozenset(
    {
        Intent.MUSIC_PAUSE,
        Intent.MUSIC_RESUME,
        Intent.MUSIC_SKIP,
        Intent.MUSIC_LOUDER,
        Intent.MUSIC_QUIETER,
    }
)

#: Есть ли в реплике вообще что-то музыкальное — триггер для LLM-фоллбэка.
_HINT = re.compile(
    r"\b(музык\w*|трек\w*|песн\w*|плейлист\w*|альбом\w*|фон\w*|звук\w*|громкост\w*"
    r"|включ\w*|постав\w*|врубн?и\w*|сыграй|играй|послуша\w*|поставь)\b"
)

#: Слова, которые в запросе на поиск трека — шум.
_FILLER = re.compile(
    r"\b(включи|включай|поставь|ставь|запусти|врубни|врубай|сыграй|играй|поищи|найди"
    r"|музыку|музыка|музыки|трек|трека|треки|песню|песня|песни|композицию|композиция"
    r"|нам|мне|пожалуйста|давай|давайте|ка|что-нибудь|чтонибудь|какую-нибудь|фоном|на фон)\b"
)

_JSON_BLOB = re.compile(r"\{.*?\}", re.DOTALL)

_CLASSIFIER_SYSTEM = (
    "Ты классификатор намерений голосового бота. Определи, чего хочет человек. "
    "Ответь ТОЛЬКО одним JSON-объектом без пояснений.\n"
    'Формат: {"intent": "<одно из: chat, silence, music_play, music_pause, music_resume, '
    'music_skip, music_stop, music_louder, music_quieter>", "query": "<название трека '
    'или пустая строка>"}\n'
    'Если человек просто разговаривает — {"intent": "chat", "query": ""}.\n'
    'Если человека просят заткнуться и замолчать — {"intent": "silence", "query": ""}.'
)

_SCHEMA = {
    "type": "object",
    "properties": {
        "intent": {"type": "string", "enum": [i.value for i in Intent]},
        "query": {"type": "string"},
    },
    "required": ["intent", "query"],
}


def extract_query(text: str) -> str:
    """Выкинуть из «поставь нам что-нибудь из Кино» всё, кроме «из Кино»."""
    cleaned = _FILLER.sub(" ", text)
    cleaned = re.sub(r"\s+", " ", cleaned).strip(" ,.-—")
    return cleaned


def classify_by_rules(text: str, *, music_playing: bool = False) -> Decision | None:
    """Быстрый путь. None — правила не сработали.

    ``music_playing`` — играет ли сейчас что-нибудь. От этого зависит разбор
    неоднозначных формулировок вроде «продолжай» или «дальше».
    """
    lowered = text.lower().replace("ё", "е")
    for intent, pattern in _RULES:
        if pattern.search(lowered):
            query = extract_query(lowered) if intent is Intent.MUSIC_PLAY else ""
            if intent is Intent.MUSIC_PLAY and not query and not _MUSIC_NOUN.search(lowered):
                # «включи» без объекта и без слова «музыка» — это не команда
                # плееру, а начало обычной фразы. Пусть разбирается диалог.
                continue
            if (
                intent in _NEEDS_MUSIC_CONTEXT
                and not music_playing
                and not _MUSIC_NOUN.search(lowered)
            ):
                continue
            return Decision(intent=intent, query=query, source="regex")
    return None


class IntentRouter:
    def __init__(self, cfg, llm=None) -> None:  # noqa: ANN001
        section = cfg.get("intent")
        self.mode = str(section.get("llm_fallback", "hinted")) if section else "hinted"
        self.llm = llm

    async def route(self, text: str, *, music_playing: bool = False) -> Decision:
        decision = classify_by_rules(text, music_playing=music_playing)
        if decision is not None:
            log.debug("intent=%s (regex) query=%r", decision.intent.value, decision.query)
            return decision

        if not self._should_ask_llm(text):
            return Decision(intent=Intent.CHAT, source="default")

        decision = await self._classify_by_llm(text)
        log.debug("intent=%s (%s) query=%r", decision.intent.value, decision.source, decision.query)
        return decision

    def _should_ask_llm(self, text: str) -> bool:
        if self.llm is None or self.mode == "never":
            return False
        if self.mode == "always":
            return True
        return bool(_HINT.search(text.lower().replace("ё", "е")))

    async def _classify_by_llm(self, text: str) -> Decision:
        try:
            with stage(log, "intent-llm"):
                raw = await self.llm.complete(
                    [
                        {"role": "system", "content": _CLASSIFIER_SYSTEM},
                        {"role": "user", "content": text},
                    ],
                    max_tokens=64,
                    temperature=0.0,
                    json_schema=_SCHEMA,
                )
        except Exception:
            # Классификатор — не критичный путь: не смог, значит просто болтаем.
            log.exception("LLM-классификатор недоступен, считаю реплику болтовнёй")
            return Decision(intent=Intent.CHAT, source="llm-error")

        return _parse_decision(raw)


def _parse_decision(raw: str) -> Decision:
    """Разобрать ответ модели, не веря ему на слово."""
    blob = _JSON_BLOB.search(raw or "")
    if blob is None:
        return Decision(intent=Intent.CHAT, source="llm-unparsed")
    try:
        data = json.loads(blob.group(0))
    except json.JSONDecodeError:
        return Decision(intent=Intent.CHAT, source="llm-unparsed")

    try:
        intent = Intent(str(data.get("intent", "chat")))
    except ValueError:
        intent = Intent.CHAT

    query = str(data.get("query", "") or "").strip()
    return Decision(intent=intent, query=query, source="llm")
