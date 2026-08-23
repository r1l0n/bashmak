"""Что человек хотел: поболтать или порулить музыкой.

Два уровня:

1. Регекс по ключевым словам — мгновенно, без LLM. Ловит подавляющее
   большинство команд («поставь Кино», «следующий трек», «сделай потише»).
2. Классификация той же LLM — для формулировок, которые правила не разобрали.

Второй уровень по умолчанию включается только при музыкальном намёке в реплике
(``llm_fallback: hinted``). Иначе каждая обычная реплика стоила бы лишнего
инференса на CPU, то есть примерно секунды к времени ответа.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from enum import Enum

from ..llm.client import INTENT_SLOT
from ..utils.logging import stage

log = logging.getLogger(__name__)


class Intent(str, Enum):
    CHAT = "chat"
    #: Команда молчания: замолчать и выключить музыку, вслух не отвечая.
    SILENCE = "silence"
    #: «Расскажи анекдот» — идёт мимо диалога, в свой источник (bashmak/jokes.py).
    JOKE = "joke"
    MUSIC_PLAY = "music_play"
    MUSIC_PAUSE = "music_pause"
    MUSIC_RESUME = "music_resume"
    MUSIC_SKIP = "music_skip"
    MUSIC_STOP = "music_stop"
    MUSIC_LOUDER = "music_louder"
    MUSIC_QUIETER = "music_quieter"
    #: «Громкость 70» — выставить значение, а не сдвинуть на шаг.
    MUSIC_VOLUME = "music_volume"


@dataclass(slots=True)
class Decision:
    intent: Intent
    query: str = ""
    source: str = "regex"
    #: Проценты для music_volume. None — команду опознали, а число нет.
    level: int | None = None


#: Круглые числа словами: Whisper пишет их то цифрами, то прописью, и «громкость
#: пятьдесят» должна работать так же, как «громкость 50». Составных чисел
#: («шестьдесят пять») в списке нет: громкость называют круглой.
_VOLUME_WORDS = {
    "ноль": 0,
    "десять": 10,
    "двадцать": 20,
    "тридцать": 30,
    "сорок": 40,
    "пятьдесят": 50,
    "шестьдесят": 60,
    "семьдесят": 70,
    "восемьдесят": 80,
    "девяносто": 90,
    "сто": 100,
}

#: «Громкость 70», «поставь громкость на 50 процентов», «громкость сто».
#: Между словом и числом \W*, а не \s*: через буквы он не переходит, поэтому
#: число привязывается именно к «громкости», а не к первому попавшемуся в фразе.
#: «Включи звуки природы 10 часов» под правило не подпадает.
_VOLUME_LEVEL = re.compile(
    r"\bгромкост\w*\W*(?:на\W+|до\W+|в\W+)?(\d{1,3}|" + "|".join(_VOLUME_WORDS) + r")\b"
)

# Порядок важен: «выключи музыку» должно попасть в stop, а не в play по «включ».
#
# Подбор слов для SILENCE. «Тише»/«потише» исключены — это громкость («сделай
# потише»). Голое «хватит» исключено — оно занято правилом ниже («хватит
# музыки»). «Заглохни» и «тишина» перенесены сюда из music_stop: по смыслу это
# «замолчи», а не «сними трек», при этом silence музыку тоже останавливает.
_RULES: tuple[tuple[Intent, re.Pattern[str]], ...] = (
    (
        Intent.SILENCE,
        re.compile(
            # Только повелительные формы: «завалил экзамен» и «молчание» —
            # обычная речь, а не команда бота.
            r"\b(завали|заткн\w*|захлопни|заглохни|тишина|уймись|замолкни"
            r"|(?:за|по)?молчи|молчать|стоп"
            r"|не пизди|хорош пиздеть|хватит (?:болтать|говорить|трещать))\b"
        ),
    ),
    # Раньше музыкальных правил: «включи анекдот» — это анекдот, а не поиск на
    # YouTube. Ловится по существительному, потому что глагол при нём любой
    # («расскажи», «давай», «зачитай») или его нет вовсе — «Башмак, анекдот».
    #
    # «Пошути» только в повелительном: \b не даёт зацепить «пошутить» из
    # обычной речи вроде «он хотел пошутить».
    (Intent.JOKE, re.compile(r"\bанекдот\w*\b|\bпошути\b")),
    (Intent.MUSIC_STOP, re.compile(r"\b(выключ\w*|останов\w*|стоп|прекрат\w*|хватит)\b.*\b(музык\w*|трек\w*|песн\w*|плейлист\w*)")),
    (Intent.MUSIC_PAUSE, re.compile(r"\b(пауз\w*|приостанов\w*|погоди с музык\w*)\b")),
    (Intent.MUSIC_RESUME, re.compile(r"\b(продолж\w*|возобнов\w*|снова играй|с паузы)\b")),
    (Intent.MUSIC_SKIP, re.compile(r"\b(следующ\w*|дальше|скип\w*|пропусти|переключ\w*|другую песн\w*|другой трек)\b")),
    # Раньше «прибавь громкость»: с числом это установка, а не шаг.
    (Intent.MUSIC_VOLUME, _VOLUME_LEVEL),
    (Intent.MUSIC_LOUDER, re.compile(r"\b(по)?громче\b|\bприбав\w* (звук|громкост\w*)\b")),
    (Intent.MUSIC_QUIETER, re.compile(r"\b(по)?тише\b|\bубав\w* (звук|громкост\w*)\b")),
    (Intent.MUSIC_PLAY, re.compile(r"\b(включ\w*|постав\w*|запусти|врубн?и\w*|сыграй|играй|поищи|найди)\b")),
)

#: Явное упоминание музыки — по нему отличаем «включи музыку» от «включи мозг».
_MUSIC_NOUN = re.compile(r"\b(музык\w*|трек\w*|песн\w*|плейлист\w*|альбом\w*|композици\w*)\b")

#: «Что-нибудь на свой вкус» — просьба выбрать за человека. Такой же повод
#: считать реплику командой плееру, как и слово «музыка»: названия в ней нет и
#: не будет, но включить просят вполне определённо.
#:
#: «ё» к этому моменту уже заменена на «е» (см. classify_by_rules), поэтому в
#: списке «че-нибудь», а не «чё-нибудь».
_ANYTHING = re.compile(
    r"\b(?:что|чего|че|чо)[\s-]?нибудь\b|\bчто[\s-]?то\b"
    r"|\bкак(?:ую|ой|ое)[\s-]?нибудь\b|\bлюб(?:ую|ой|ое)\b|\bрандом\w*"
    r"|\bна свой вкус\b|\bсам\w*\s+(?:выбер|реши)\w*"
)

#: Намерения, слова которых часто встречаются в обычной речи: «Башмак,
#: продолжай» — не resume, «а что дальше?» — не skip, «говори потише» — не
#: громкость. Такие команды принимаются, только если музыка названа явно или
#: уже играет.
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

#: Слова, которые в запросе на поиск трека являются шумом.
#:
#: Обороты «что-нибудь», «любой» и «на свой вкус» тоже шум, и вычищаются они не
#: для красоты: без этого «включи любой трек» ушло бы искать на YouTube слово
#: «любой». Пустой остаток — это и есть команда «выбери сам».
#: Многословные варианты стоят раньше однословных: альтернатива выбирается
#: первой подошедшей, и «на» из «на фон» съело бы начало «на свой вкус».
_FILLER = re.compile(
    r"\b(на свой вкус|сам\w* (?:выбер|реши)\w*|(?:выбер|реши)\w* сам\w*"
    r"|включи|включай|поставь|ставь|запусти|врубни|врубай|сыграй|играй"
    r"|поищи|найди|музыку|музыка|музыки|трек|трека|треки|песню|песня|песни"
    r"|композицию|композиция|нам|мне|пожалуйста|давай|давайте|ка"
    r"|что-нибудь|что нибудь|чтонибудь|чего-нибудь|чего нибудь|че-нибудь|че нибудь"
    r"|что-то|что то|чтото|какую-нибудь|какую нибудь|какой-нибудь|какой нибудь"
    r"|какое-нибудь|какое нибудь|любую|любой|любое|рандомную|рандомный|рандомное|рандом"
    r"|фоном|на фон)\b"
)

_JSON_BLOB = re.compile(r"\{.*?\}", re.DOTALL)

_CLASSIFIER_SYSTEM = (
    "Ты классификатор намерений голосового бота. Определи, чего хочет человек. "
    "Ответь ТОЛЬКО одним JSON-объектом без пояснений.\n"
    'Формат: {"intent": "<одно из: chat, silence, music_play, music_pause, music_resume, '
    'music_skip, music_stop, music_louder, music_quieter, music_volume>", "query": '
    '"<название трека, число процентов или пустая строка>"}\n'
    'Если человек просто разговаривает — {"intent": "chat", "query": ""}.\n'
    'Если человека просят заткнуться и замолчать — {"intent": "silence", "query": ""}.\n'
    'Если просят музыку, но названия не назвали — {"intent": "music_play", "query": ""}; '
    "название не выдумывай, пустой запрос значит «бот выберет сам».\n"
    'Если названа конкретная громкость — {"intent": "music_volume", "query": "<число>"}; '
    'если просят просто громче или тише — music_louder или music_quieter.'
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


def extract_level(text: str) -> int | None:
    """Достать проценты из «громкость 70». None — числа в реплике нет."""
    found = _VOLUME_LEVEL.search(text.lower().replace("ё", "е"))
    if found is None:
        # Классификатор отдаёт число отдельным полем, без слова «громкость».
        found = re.fullmatch(r"\s*(\d{1,3})\s*", text)
        return int(found.group(1)) if found else None
    value = found.group(1)
    return int(value) if value.isdigit() else _VOLUME_WORDS[value]


def classify_by_rules(text: str, *, music_playing: bool = False) -> Decision | None:
    """Быстрый путь. None — правила не сработали.

    ``music_playing`` — играет ли сейчас что-нибудь. От этого зависит разбор
    неоднозначных формулировок вроде «продолжай» или «дальше».
    """
    lowered = text.lower().replace("ё", "е")
    for intent, pattern in _RULES:
        if pattern.search(lowered):
            query = extract_query(lowered) if intent is Intent.MUSIC_PLAY else ""
            if (
                intent is Intent.MUSIC_PLAY
                and not query
                and not _MUSIC_NOUN.search(lowered)
                and not _ANYTHING.search(lowered)
            ):
                # «включи» без объекта, без слова «музыка» и без «что-нибудь» —
                # не команда плееру, а начало обычной фразы. Уходит в диалог.
                continue
            if (
                intent in _NEEDS_MUSIC_CONTEXT
                and not music_playing
                and not _MUSIC_NOUN.search(lowered)
            ):
                continue
            level = extract_level(lowered) if intent is Intent.MUSIC_VOLUME else None
            return Decision(intent=intent, query=query, source="regex", level=level)
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
                    # Свой слот: системная часть у классификатора своя, и в
                    # общем слоте она вытирала кеш префикса диалога.
                    slot=INTENT_SLOT,
                )
        except Exception:
            # Классификатор не критичен: не сработал — реплика идёт в диалог.
            log.exception("LLM-классификатор недоступен, реплика уходит в диалог")
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
    level = extract_level(query) if intent is Intent.MUSIC_VOLUME else None
    return Decision(intent=intent, query=query, source="llm", level=level)
