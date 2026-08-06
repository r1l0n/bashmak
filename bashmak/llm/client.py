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


def render_prompt(messages: list[dict[str, str]]) -> str:
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


#: Сколько символов пояснения от сервера писать в лог. Ошибка шаблона чата
#: бывает длинной (minja печатает кусок шаблона), но информативно её начало.
_DETAIL_CHARS = 600


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
        self.max_tokens = int(cfg.get("max_tokens", 200))
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
        # Без этого его запрос вставал бы в очередь уже на самом llama-server
        # (--parallel 1), и request_timeout_s тикал бы всё это ожидание.
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
        retries: int = 1,
    ) -> str:
        self._check_roles(messages)
        prompt = render_prompt(messages)
        payload: dict[str, Any] = {
            "prompt": prompt,
            "n_predict": max_tokens if max_tokens is not None else self.max_tokens,
            "temperature": temperature if temperature is not None else self.temperature,
            "top_p": self.top_p,
            "stream": False,
            # Конец реплики — он же EOS, но модель иногда печатает его текстом.
            "stop": [MESSAGE_SEP],
            # Системная часть промпта не меняется, и её префикс сервер переиспользует.
            "cache_prompt": True,
        }
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
                    return (data["content"] or "").strip()
                except (httpx.HTTPError, LlmError, KeyError, IndexError, ValueError) as exc:
                    last_error = exc
                    if attempt < retries:
                        log.warning("LLM-запрос не удался (%s), повтор %d", exc, attempt + 1)
                        await asyncio.sleep(1.0)

        raise LlmError(f"llama-server не ответил: {last_error}") from last_error
