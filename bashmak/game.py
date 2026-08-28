"""Запуск игры по кодовой фразе: «Башмак, дай новый сценарий».

За фразой стоит HTTP-ручка на машине с игрой: переход по ссылке поднимает её
там. Боту остаётся сходить по ссылке и отчитаться условленным ответом —
кодовая фраза и ответ на неё часть шутки, а не описание происходящего.

Ссылка с токеном в шаблоне конфига не лежит: она даёт запуск на чужой машине
любому, кто её увидит, а config.example.yaml уезжает в репозиторий. Настройка —
в своём config.yaml (``game.url``) или в окружении (``BASHMAK_GAME_URL``), как
и токен Discord.
"""

from __future__ import annotations

import logging
import os
import time

import httpx

log = logging.getLogger(__name__)

#: Что бот говорит, когда сценарий поднялся.
_DEFAULT_REPLY = "Иван Михайлович написал парочку за время вашего отсутствия"

#: Сколько ждать ответа. На ответ бот всё равно не смотрит (см. launch), так что
#: это не срок ожидания, а длина паузы, в которую он молчит в канале. Запрос
#: уходит на соединении, задолго до её конца.
_DEFAULT_TIMEOUT = 3.0

#: Переменная окружения со ссылкой — перекрывает конфиг (см. .env.example).
ENV_URL = "BASHMAK_GAME_URL"


class GameLauncher:
    """Дёргает ручку запуска.

    Живёт на боте, а не на голосовой сессии: ручка одна на все каналы, и
    держать под неё по клиенту на канал незачем.
    """

    def __init__(self, cfg) -> None:  # noqa: ANN001 — bashmak.config.Config
        section = cfg.get("game")
        url = str(section.get("url", "") or "") if section else ""
        self._url = os.environ.get(ENV_URL, "").strip() or url.strip()
        self._timeout = float(section.get("timeout_s", _DEFAULT_TIMEOUT)) if section else _DEFAULT_TIMEOUT
        # Пустой reply в конфиге — это не «молчать», а недописанная строка:
        # берём условленный ответ.
        self._reply = str(section.get("reply", "") or _DEFAULT_REPLY) if section else _DEFAULT_REPLY
        self._client: httpx.AsyncClient | None = None

    @property
    def configured(self) -> bool:
        """Есть ли куда идти. False — фраза распознаётся, но запускать нечего."""
        return bool(self._url)

    async def launch(self) -> str:
        """Сходить по ссылке. Возвращает фразу, которую бот скажет вслух.

        По ответу ручки об успехе не судим: она сначала поднимает игру и только
        потом отчитывается, так что 500 приходит уже поверх запущенной, а иногда
        ответа нет вовсе. Проверять тут нечего — вслух звучит условленная фраза
        при любом исходе, а что было на самом деле, остаётся в логе.

        Наружу не бросает: обработчик реплики ждёт готовый ответ, а не исключение.
        """
        if not self._url:
            # Единственный случай, когда фраза не звучит: идти некуда, и это
            # не сбой запуска, а недонастроенный бот.
            log.warning("ссылка на запуск не настроена (game.url или %s)", ENV_URL)
            return "Не могу: сценарии лежат не у меня. Пропиши ссылку в конфиге."

        started = time.monotonic()
        try:
            response = await self._http().get(self._url)
        except Exception as exc:
            # Вслух не выносится и это, но в логе выключенная машина должна
            # отличаться от ответившей: тут запрос не ушёл вообще.
            log.warning(
                "до ручки запуска не достучался за %.1f с (%s: %s)",
                time.monotonic() - started,
                type(exc).__name__,
                exc,
            )
        else:
            log.info(
                "ручка запуска ответила %s за %.1f с",
                response.status_code,
                time.monotonic() - started,
            )
        return self._reply

    async def close(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    def _http(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=self._timeout, follow_redirects=True)
        return self._client
