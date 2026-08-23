"""Анекдоты: «Башмак, расскажи анекдот».

Берутся из чужой RSS-ленты, а не из головы модели. Анекдот держится на
неожиданной концовке, и 8B-lite выдаёт вместо неё присказку: форма похожа,
смешного нет. Модель остаётся фолбэком — на случай, когда сети нет или
источник лёг.

Лента отдаёт десяток анекдотов за запрос, поэтому она кешируется: просьба
рассказать — это следующий анекдот из уже разобранной пачки, а не поход в
сеть. Рассказанные помнятся по guid, чтобы подряд не повторяться.

Текст не фильтруется: что пришло, то и звучит. Бот и так матерится по промпту,
и отдельная планка «прилично/неприлично» здесь никому не нужна.
"""

from __future__ import annotations

import asyncio
import html
import logging
import random
import re
import time
import xml.etree.ElementTree as ET
from collections import deque
from dataclasses import dataclass

import httpx

from .llm.persona import clean_reply

log = logging.getLogger(__name__)

#: Лента анекдотов. Отдаёт десяток свежих записей, обновляется раз в сутки.
_DEFAULT_URL = "https://www.anekdot.ru/rss/export_j.xml"

#: Свой User-Agent: на голом httpx часть сайтов отвечает 403.
_USER_AGENT = "Mozilla/5.0 (compatible; bashmak-bot)"

#: Сколько рассказанных помнить, чтобы не повторяться. Заметно больше пачки:
#: за время жизни кеша лента отдаёт одно и то же, и памяти в десяток записей
#: хватило бы ровно на один круг.
RECENT_LIMIT = 60

#: Анекдот длиннее одной фразы, и лимит диалога (несколько десятков токенов)
#: обрезал бы его на середине — у фолбэка свой.
_LLM_MAX_TOKENS = 200

#: Промпт фолбэка. Отдельный от персоны: там «одним предложением», а анекдот в
#: одно предложение не укладывается.
_LLM_PROMPT = (
    "Ты рассказываешь анекдоты в компании своих. Выдай один короткий "
    "матерный анекдот и ничего больше: без вступления, без пояснений, без "
    "«вот анекдот»."
)

#: Перевод строки в ленте размечен <br>, остальная разметка — мусор от сайта.
_BR = re.compile(r"<br\s*/?>", re.IGNORECASE)
_TAG = re.compile(r"<[^>]+>")
#: Пробелы внутри строки. Парный тег («<b>он</b>») после вычистки оставляет два
#: подряд, и склеивать их обратно приходится отдельно.
_SPACES = re.compile(r"[^\S\n]+")


class JokeError(RuntimeError):
    pass


@dataclass(slots=True)
class Joke:
    text: str
    #: guid из ленты — по нему отсеиваются уже рассказанные.
    uid: str


def _plain_text(raw: str) -> str:
    """Из HTML ленты — текст для синтезатора.

    Теги Silero не пропускает мимо, а читает вслух, поэтому вычищаются все.
    Пустые строки убираются тоже: синтез режет текст по предложениям, и лишний
    перевод строки — это пауза на ровном месте.
    """
    text = _TAG.sub(" ", _BR.sub("\n", raw))
    lines = (_SPACES.sub(" ", line).strip() for line in html.unescape(text).splitlines())
    return "\n".join(line for line in lines if line).strip()


def _parse(xml: str) -> list[Joke]:
    """Разобрать ленту. Битый XML — это ошибка, пустая лента — просто пусто."""
    try:
        root = ET.fromstring(xml)
    except ET.ParseError as exc:
        raise JokeError(f"лента не разобралась: {exc}") from exc

    jokes: list[Joke] = []
    for item in root.iter("item"):
        text = _plain_text(item.findtext("description") or "")
        if not text:
            continue
        # guid есть не в каждой ленте; ссылка и сам текст — запасные ключи.
        uid = (item.findtext("guid") or item.findtext("link") or text).strip()
        jokes.append(Joke(text=text, uid=uid[:200]))
    return jokes


class JokeTeller:
    """Источник анекдотов с кешем и памятью на рассказанное.

    Живёт на боте, а не на голосовой сессии: пачка из ленты и список
    рассказанного общие для всех каналов — иначе в соседнем канале звучал бы
    тот же анекдот, что минуту назад в этом.
    """

    def __init__(self, cfg, llm=None) -> None:  # noqa: ANN001
        section = cfg.get("jokes")
        self._url = str(section.get("url", _DEFAULT_URL)) if section else _DEFAULT_URL
        self._timeout = float(section.get("timeout_s", 10)) if section else 10.0
        self._ttl = float(section.get("cache_ttl_s", 900)) if section else 900.0
        limit = int(section.get("recent_limit", RECENT_LIMIT)) if section else RECENT_LIMIT

        self._llm = llm
        self._pool: list[Joke] = []
        self._fetched_at = 0.0
        self._recent: deque[str] = deque(maxlen=limit)
        self._client: httpx.AsyncClient | None = None
        # Две просьбы подряд не должны тянуть ленту дважды.
        self._lock = asyncio.Lock()

    async def tell(self) -> str:
        """Анекдот вслух. Возвращает готовую фразу, наружу не бросает.

        Порядок: пачка из ленты, потом модель, потом честное «нечем».
        """
        async with self._lock:
            joke = await self._from_feed()
        if joke:
            return joke

        joke = await self._from_llm()
        if joke:
            log.info("анекдот от модели: лента недоступна")
            return joke

        return "Анекдоты кончились, а интернет сдох. Сам чего-нибудь расскажи."

    async def close(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    # ----------------------------------------------------------- внутрь ---
    def _http(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(
                timeout=self._timeout,
                follow_redirects=True,
                headers={"User-Agent": _USER_AGENT},
            )
        return self._client

    async def _from_feed(self) -> str:
        """Следующий анекдот из пачки, подтянув её при надобности.

        За свежей пачкой идём и когда кеш протух, и когда её рассказали
        целиком: во втором случае повтор иначе прозвучал бы прямо сейчас, а
        обновление досталось бы только следующей просьбе.
        """
        if self._stale() or not self._unheard():
            await self._refresh()

        # Пачку рассказали, а свежей не принесли (лента не обновилась или
        # недоступна) — второй круг лучше молчания.
        fresh = self._unheard() or self._pool
        if not fresh:
            return ""

        joke = random.choice(fresh)
        self._recent.append(joke.uid)
        return joke.text

    def _stale(self) -> bool:
        return not self._pool or time.monotonic() - self._fetched_at > self._ttl

    def _unheard(self) -> list[Joke]:
        return [joke for joke in self._pool if joke.uid not in self._recent]

    async def _refresh(self) -> None:
        """Подтянуть пачку. Неудача — не беда: прежняя уже разобрана и лежит."""
        try:
            self._pool = await self._fetch()
            self._fetched_at = time.monotonic()
        except Exception as exc:
            log.warning("лента анекдотов недоступна (%s)", exc)

    async def _fetch(self) -> list[Joke]:
        response = await self._http().get(self._url)
        response.raise_for_status()
        jokes = _parse(response.text)
        log.debug("лента анекдотов: %d записей", len(jokes))
        return jokes

    async def _from_llm(self) -> str:
        """Фолбэк: пусть сочинит сама.

        Без явного слота: фолбэк редкий, а обоих слотов llama-server заняты
        диалогом и классификатором (см. client.CHAT_SLOT). Отдать один из них
        под анекдот — значит выбить чужой кеш префикса ради разового запроса;
        сервер и сам выберет слот по длиннейшему общему префиксу.
        """
        if self._llm is None:
            return ""
        try:
            raw = await self._llm.complete(
                [
                    {"role": "system", "content": _LLM_PROMPT},
                    {"role": "user", "content": "Расскажи анекдот."},
                ],
                max_tokens=_LLM_MAX_TOKENS,
            )
        except Exception as exc:
            log.warning("модель анекдот не выдала (%s)", exc)
            return ""
        # Та же чистка, что и у обычной реплики: модель дописывает за
        # собеседника и лепит разметку одинаково везде.
        return clean_reply(raw)
