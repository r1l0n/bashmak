"""HTTP-клиент к llama-server (OpenAI-совместимый API)."""

from __future__ import annotations

import asyncio
import logging
from typing import Any

import httpx

log = logging.getLogger(__name__)


class LlmError(RuntimeError):
    pass


#: Разделители формата GigaChat: между ролью и текстом, и в конце сообщения.
#: <|message_sep|> у этой модели заодно EOS — по нему генерация и кончается.
ROLE_SEP = "<|role_sep|>"
MESSAGE_SEP = "<|message_sep|>"

#: Конец хода у Vikhr-YandexGPT. Тоже EOS, роль <|message_sep|> в своём формате.
TURN_END = "</s>"

#: Слоты llama-server: у каждого свой кеш префикса промпта.
#:
#: Диалог и классификатор намерений ходят в одну модель, но с совершенно
#: разными системными частями. В одном слоте они вытирали кеш друг другу:
#: после разбора «включи что-нибудь фоном» следующая обычная реплика платила
#: полным префиллом персоны — сотни токенов на CPU. Требует --parallel 2 в
#: scripts/run_llama_server.sh.
CHAT_SLOT = 0
INTENT_SLOT = 1


def _render_gigachat(messages: list[dict[str, str]]) -> str:
    """Собрать промпт в формате GigaChat.

    Вручную, а не через /v1/chat/completions: чат-слой llama.cpp этот формат не
    поддерживает. Штатный шаблон берёт разделители из переменной
    additional_special_tokens, которой в llama.cpp нет, и промпт уходит без
    разделителей вовсе. Со своим шаблоном сборка b10237 выводит peg-парсер
    ответа и падает с 500 «output does not match the expected peg-native
    format» на обычной реплике модели.

    ``<s>`` не добавляется: /completion токенизирует промпт с add_special=true
    и ставит BOS сам, второй BOS сбил бы модель.
    """
    parts: list[str] = []
    for message in messages:
        content = message["content"]
        if message["role"] == "system":
            parts.append(f"{content}{MESSAGE_SEP}")
            continue
        parts.append(f"{message['role']}{ROLE_SEP}{content}{MESSAGE_SEP}")
        if message["role"] == "user":
            # Блок функций модель ждёт после каждой реплики пользователя.
            # Пустой список — штатное обозначение «функций нет».
            parts.append(f"available functions{ROLE_SEP}[]{MESSAGE_SEP}")
    parts.append(f"assistant{ROLE_SEP}")
    return "".join(parts)


def _render_vikhr(messages: list[dict[str, str]]) -> str:
    """Собрать промпт в формате Vikhr-YandexGPT.

    Шаблон обычный, роль отдельной строкой и системная часть своим блоком::

        system\\n{системная часть}</s>\\n<s>user\\n{реплика}</s>\\n<s>assistant\\n

    Своя системная роль здесь важнее, чем кажется: она стоит первой и целиком
    попадает в кеш префикса llama-server. У самой YandexGPT-5-Lite системного
    блока нет, там инструкции пришлось бы вклеивать в первую реплику
    пользователя — то есть в тот ход, который первым и вытесняется из истории,
    и персона пересчитывалась бы заново после каждого забытого хода.

    ``<s>`` тут разделитель каждого хода, а не только начало промпта, но первый
    всё равно не пишется — BOS сервер ставит сам (см. :func:`_render_gigachat`).
    """
    parts: list[str] = []
    for index, message in enumerate(messages):
        opener = "" if index == 0 else "<s>"
        parts.append(f"{opener}{message['role']}\n{message['content']}{TURN_END}\n")
    parts.append("<s>assistant\n")
    return "".join(parts)


#: Сборщик промпта по имени формата (llm.prompt_format).
#:
#: Разделители ролей у моделей разные, а промпт мы собираем сами, мимо чат-слоя
#: llama.cpp. С чужим форматом модель не падает, а начинает дописывать диалог за
#: всех участников — persona.clean_reply() это ловит, но ответ уже испорчен.
_RENDERERS = {"gigachat": _render_gigachat, "vikhr": _render_vikhr}

#: Чем модель заканчивает ход. Он же первый стоп-маркер генерации: модели
#: случается напечатать свой EOS текстом, а не выдать его токеном.
_TURN_ENDS = {"gigachat": MESSAGE_SEP, "vikhr": TURN_END}


def render_prompt(messages: list[dict[str, str]], fmt: str = "gigachat") -> str:
    """Собрать промпт под формат модели (см. :data:`_RENDERERS`)."""
    renderer = _RENDERERS.get(fmt)
    if renderer is None:
        raise LlmError(
            f"неизвестный формат промпта {fmt!r} (есть: {', '.join(sorted(_RENDERERS))})"
        )
    return renderer(messages)


#: Стоп-строки «конец первого предложения» — для complete(one_sentence=True).
#:
#: Персона отвечает одним предложением, и всё, что модель допишет после него,
#: persona.clean_reply() всё равно выбросит. Но посчитано оно к этому моменту
#: уже будет: на CPU это лишние секунды в каждой реплике, поэтому генерацию
#: обрываем на сервере, а не после.
#:
#: Знак с пробелом, а не голый: «.» сработал бы внутри «3.5» и «т.д.», а «. » —
#: только там, где за предложением правда идёт продолжение. Последнее
#: предложение заканчивается на EOS и под правило не подпадает.
_SENTENCE_STOPS = (". ", "! ", "? ", "… ")

#: Чем сервер мог оборвать предложение — по этому знаку его и восстанавливаем.
_SENTENCE_MARKS = frozenset(".!?…")

#: Сколько символов пояснения от сервера писать в лог. Ошибка шаблона чата
#: бывает длинной (minja печатает кусок шаблона), но информативно её начало.
_DETAIL_CHARS = 600


def _answer(data: dict[str, Any]) -> str:
    """Текст ответа с возвращённым знаком конца предложения.

    Сработавшую стоп-строку сервер в ``content`` не включает: от «Иди нахер! И
    дверь закрой.» пришло бы «Иди нахер» без знака, и Silero прочитал бы фразу
    с оборванной интонацией. Знак берём из ``stopping_word``.

    Поле есть не во всех сборках llama.cpp, и приходить может как «! », так и
    «!». Нет поля или знак незнакомый — остаёмся без знака: интонация того не
    стоит, чтобы падать.
    """
    text = (data["content"] or "").rstrip()
    stopped_by = str(data.get("stopping_word") or "").strip()
    if text and stopped_by in _SENTENCE_MARKS:
        text += stopped_by
    return text.strip()


def _server_error(response: httpx.Response) -> str:
    """Достать из ответа причину, а не только код.

    raise_for_status() показывает «500 Internal Server Error» и отбрасывает
    тело, а llama-server пишет в нём, что именно случилось: ошибку шаблона
    чата, переполненный контекст, неизвестный параметр. Поэтому тело
    разбирается вручную.
    """
    detail = ""
    try:
        body = response.json()
        if isinstance(body, dict):
            error = body.get("error")
            detail = (error.get("message") if isinstance(error, dict) else str(error)) or ""
    except ValueError:
        pass
    detail = (detail or response.text).strip()[:_DETAIL_CHARS]
    return f"{response.status_code} от llama-server: {detail or '(пустой ответ)'}"


class LlmClient:
    def __init__(self, cfg) -> None:  # noqa: ANN001 — bashmak.config.Section
        self.base_url = str(cfg.get("server_url", "http://127.0.0.1:8080")).rstrip("/")
        # Проверяем здесь, а не при сборке промпта: опечатка в prompt_format
        # иначе всплыла бы только на первой реплике, уже в голосовом канале.
        self.prompt_format = str(cfg.get("prompt_format", "gigachat"))
        if self.prompt_format not in _RENDERERS:
            raise LlmError(
                f"llm.prompt_format={self.prompt_format!r} в конфиге — "
                f"такого формата нет (есть: {', '.join(sorted(_RENDERERS))})"
            )
        self.max_tokens = int(cfg.get("max_tokens", 48))
        self.temperature = float(cfg.get("temperature", 0.7))
        self.top_p = float(cfg.get("top_p", 0.9))
        timeout = float(cfg.get("request_timeout_s", 120))

        # Один инференс за раз всё равно (см. queue_manager), пул соединений не нужен.
        self._client = httpx.AsyncClient(
            base_url=self.base_url,
            timeout=httpx.Timeout(timeout, connect=10.0),
            limits=httpx.Limits(max_connections=4),
        )
        # Инвариант «один инференс за раз» держится здесь, а не только в
        # очереди: мимо очереди в LLM ходит ещё и intent-классификатор.
        #
        # Слотов у сервера два (--parallel 2), но это ради двух кешей префикса,
        # а не ради параллельного счёта: на CPU два инференса одной модели
        # делят те же ядра и кеш и оба идут медленнее, чем шли бы по очереди.
        # Так что оба слота обслуживаются по-прежнему строго последовательно.
        self._inference = asyncio.Lock()

    async def close(self) -> None:
        await self._client.aclose()

    async def health(self) -> bool:
        """Готов ли сервер принимать запросы (модель уже в памяти)."""
        try:
            response = await self._client.get("/health", timeout=5.0)
        except httpx.HTTPError as exc:
            log.debug("llama-server недоступен: %s", exc)
            return False
        return response.status_code == 200

    async def wait_until_ready(self, timeout: float = 300.0, interval: float = 3.0) -> bool:
        """Модель в 12 ГБ грузится с диска минутами, бот должен её дождаться."""
        deadline = asyncio.get_running_loop().time() + timeout
        announced = False
        while asyncio.get_running_loop().time() < deadline:
            if await self.health():
                return True
            if not announced:
                log.info("жду llama-server на %s ...", self.base_url)
                announced = True
            await asyncio.sleep(interval)
        return False

    @staticmethod
    def _check_roles(messages: list[dict[str, str]]) -> None:
        """Шаблон GigaChat требует строгого чередования user/assistant.

        Нарушение он ловит сам, но уже на сервере: отдаёт 500 с ошибкой Jinja,
        и в логе бота остаётся только «llama-server не ответил». Здесь видно,
        кто именно собрал такой список.
        """
        roles = [m["role"] for m in messages if m["role"] != "system"]
        expected = ["user" if i % 2 == 0 else "assistant" for i in range(len(roles))]
        if roles != expected:
            raise LlmError(f"роли должны идти user/assistant по очереди, а пришло: {roles}")

    async def complete(
        self,
        messages: list[dict[str, str]],
        *,
        max_tokens: int | None = None,
        temperature: float | None = None,
        json_schema: dict[str, Any] | None = None,
        one_sentence: bool = False,
        slot: int | None = None,
        retries: int = 1,
    ) -> str:
        """Сходить в модель и вернуть текст ответа.

        ``one_sentence`` — оборвать генерацию на конце первого предложения (см.
        :data:`_SENTENCE_STOPS`). Включать только там, где длиннее одного
        предложения ответ и не нужен: для JSON-классификатора это порезало бы
        объект пополам на запросе вроде «Кино. Группа крови».

        ``slot`` — в каком слоте llama-server считать. У слота свой кеш
        префикса, а промпты у диалога и у классификатора намерений разные:
        деля один слот, они вытирали кеш друг другу, и следующий запрос платил
        полным префиллом системной части. Сервер и сам выбирает слот по
        длиннейшему общему префиксу, так что это подстраховка; сборка, которая
        поля не знает, просто его игнорирует.
        """
        self._check_roles(messages)
        prompt = render_prompt(messages, self.prompt_format)
        # Конец реплики — он же EOS, но модель иногда печатает его текстом.
        stop = [_TURN_ENDS[self.prompt_format], *(_SENTENCE_STOPS if one_sentence else ())]
        payload: dict[str, Any] = {
            "prompt": prompt,
            "n_predict": max_tokens if max_tokens is not None else self.max_tokens,
            "temperature": temperature if temperature is not None else self.temperature,
            "top_p": self.top_p,
            "stream": False,
            "stop": stop,
            # Сервер переиспользует общий префикс с тем, что уже лежит в слоте,
            # и разбирает заново только хвост после первого отличия.
            "cache_prompt": True,
        }
        if slot is not None:
            payload["id_slot"] = slot
        if json_schema is not None:
            # llama-server умеет ограничивать вывод схемой, тогда JSON приходит
            # гарантированно валидным. Если сборка старая и параметр не понят,
            # ответ всё равно разбирается защитно на стороне вызывающего.
            payload["json_schema"] = json_schema

        log.debug("запрос к LLM: %r", prompt)

        last_error: Exception | None = None
        async with self._inference:
            for attempt in range(retries + 1):
                try:
                    response = await self._client.post("/completion", json=payload)
                    if response.status_code >= 400:
                        raise LlmError(_server_error(response))
                    data = response.json()
                    # Без этого «llm 5.8 с» не отличить от «модель досчитала до
                    # n_predict, а всё после первого предложения выбросил
                    # clean_reply»: время одинаковое, лечится разным.
                    log.debug(
                        "LLM: %s/%s токенов, промпт %s, стоп %s",
                        data.get("tokens_predicted"),
                        payload["n_predict"],
                        data.get("tokens_evaluated"),
                        "лимит"
                        if data.get("stopped_limit")
                        else "стоп-строка"
                        if data.get("stopped_word")
                        else "eos",
                    )
                    return _answer(data)
                except (httpx.HTTPError, LlmError, KeyError, IndexError, ValueError) as exc:
                    last_error = exc
                    if attempt < retries:
                        log.warning("LLM-запрос не удался (%s), повтор %d", exc, attempt + 1)
                        await asyncio.sleep(1.0)

        raise LlmError(f"llama-server не ответил: {last_error}") from last_error
