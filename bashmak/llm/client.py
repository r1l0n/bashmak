"""HTTP-клиент к llama-server (OpenAI-совместимый API)."""

from __future__ import annotations

import asyncio
import logging
from typing import Any

import httpx

log = logging.getLogger(__name__)


class LlmError(RuntimeError):
    pass


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
        """Модель в 12 ГБ грузится с диска минуты — бот должен её дождаться, а не упасть."""
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

        Нарушение он ловит сам — но уже на сервере, отдавая 500 с жалобой из
        Jinja, и в логе бота остаётся только «llama-server не ответил».
        Здесь же видно, кто именно собрал такой список.
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
        payload: dict[str, Any] = {
            "messages": messages,
            "max_tokens": max_tokens if max_tokens is not None else self.max_tokens,
            "temperature": temperature if temperature is not None else self.temperature,
            "top_p": self.top_p,
            "stream": False,
        }
        if json_schema is not None:
            # llama-server умеет ограничивать вывод схемой — так JSON приходит
            # валидным, а не «почти». Если сборка старая и параметр не понят,
            # ответ всё равно разбирается защитно на стороне вызывающего.
            payload["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": "decision", "schema": json_schema, "strict": True},
            }

        log.debug("запрос к LLM: %s", payload["messages"])

        last_error: Exception | None = None
        async with self._inference:
            for attempt in range(retries + 1):
                try:
                    response = await self._client.post("/v1/chat/completions", json=payload)
                    response.raise_for_status()
                    data = response.json()
                    return (data["choices"][0]["message"]["content"] or "").strip()
                except (httpx.HTTPError, KeyError, IndexError, ValueError) as exc:
                    last_error = exc
                    if attempt < retries:
                        log.warning("LLM-запрос не удался (%s), повтор %d", exc, attempt + 1)
                        await asyncio.sleep(1.0)

        raise LlmError(f"llama-server не ответил: {last_error}") from last_error
