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

#: Сколько ждать ответа. Ручка отвечать не обязана — она поднимает игру и
#: молчит, — а молчание в ответ считается успехом (см. launch). То есть это не
#: срок ожидания, а длина паузы, в которую бот молчит в канале: запрос
#: доставляется на соединении, задолго до её конца.
#:
#: Совсем убрать паузу нельзя: за неё успевает прийти отказ в соединении, и
#: выключенная машина отличается от запущенной игры именно так.
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

        Наружу не бросает: обработчик реплики ждёт готовый ответ, а не исключение.
        """
        if not self._url:
            log.warning("ссылка на запуск не настроена (game.url или %s)", ENV_URL)
            return "Не могу: сценарии лежат не у меня. Пропиши ссылку в конфиге."

        started = time.monotonic()
        try:
            response = await self._http().get(self._url)
            response.raise_for_status()
        except (httpx.ReadTimeout, httpx.WriteTimeout, httpx.ReadError, httpx.RemoteProtocolError) as exc:
            # Запрос ушёл и был принят — не дождались только ответа. Для этой
            # ручки штатно: она сначала поднимает игру, а отвечает уже потом,
            # если вообще отвечает. Игра при этом запускается, поэтому отвечаем
            # как при успехе, а странность остаётся в логе.
            #
            # Отличается от ветки ниже принципиально: там соединение не
            # состоялось, то есть до машины с игрой не дошло ничего.
            log.warning(
                "ручка запуска приняла запрос, но не ответила за %.1f с (%s) — "
                "считаю, что игра поднялась",
                time.monotonic() - started,
                type(exc).__name__,
            )
            return self._reply
        except httpx.HTTPStatusError as exc:
            # Отдельно от прочих: у HTTPStatusError в тексте вся ссылка
            # целиком — вместе с токеном, а лог пишется в файл.
            log.warning("ручка запуска ответила %s", exc.response.status_code)
            return "Сценарий не открылся, посмотри логи."
        except Exception as exc:
            log.warning("до ручки запуска не достучался (%s: %s)", type(exc).__name__, exc)
            return "Сценарий не открылся, посмотри логи."

        log.info("сценарий запущен (ответ %s за %.1f с)", response.status_code, time.monotonic() - started)
        return self._reply

    async def close(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    def _http(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=self._timeout, follow_redirects=True)
        return self._client
