"""Запуск игры по кодовой фразе: настройки и поведение ручки. Без сети."""

from __future__ import annotations

import asyncio

import httpx
import pytest

from bashmak.game import ENV_URL, GameLauncher

URL = "http://host:5000/control/dota?token=secret"


class FakeConfig:
    """cfg.get('game') — единственное, что читает GameLauncher."""

    def __init__(self, **game):
        self._section = FakeSection(game)

    def get(self, name, default=None):
        return self._section if name == "game" else default


class FakeSection:
    def __init__(self, values):
        self._values = values

    def get(self, name, default=None):
        return self._values.get(name, default)


class Empty:
    """Конфиг без секции game — так выглядит старый config.yaml."""

    def get(self, name, default=None):
        return default


class FakeResponse:
    def __init__(self, status_code=200, error=None):
        self.status_code = status_code
        self._error = error

    def raise_for_status(self):
        if self._error is not None:
            raise self._error


class FakeClient:
    def __init__(self, response=None, error=None):
        self.response = response if response is not None else FakeResponse()
        self.error = error
        self.calls: list[str] = []

    async def get(self, url):
        self.calls.append(url)
        if self.error is not None:
            raise self.error
        return self.response


def status_error(code: int) -> httpx.HTTPStatusError:
    """Ровно то, что бросает raise_for_status(): в тексте вся ссылка с токеном."""
    request = httpx.Request("GET", URL)
    return httpx.HTTPStatusError(
        f"{code}", request=request, response=httpx.Response(code, request=request)
    )


@pytest.fixture()
def loop():
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


def build(monkeypatch, client=None, **settings):
    """Пускатель с подменённой сетью: _http() отдаёт фальшивый клиент."""
    monkeypatch.delenv(ENV_URL, raising=False)
    launcher = GameLauncher(FakeConfig(**settings))
    client = client if client is not None else FakeClient()
    monkeypatch.setattr(launcher, "_http", lambda: client)
    return launcher, client


# -------------------------------------------------------------- запуск ----
def test_launch_answers_with_the_agreed_phrase(loop, monkeypatch):
    launcher, _ = build(monkeypatch, url=URL)

    assert loop.run_until_complete(launcher.launch()) == (
        "Иван Михайлович написал парочку за время вашего отсутствия"
    )


def test_launch_goes_to_the_configured_url(loop, monkeypatch):
    launcher, client = build(monkeypatch, url=URL)

    loop.run_until_complete(launcher.launch())

    assert client.calls == [URL]


def test_reply_can_be_changed_in_the_config(loop, monkeypatch):
    launcher, _ = build(monkeypatch, url=URL, reply="Готово")

    assert loop.run_until_complete(launcher.launch()) == "Готово"


# -------------------------------------------------------------- отказы ----
def test_without_a_url_nothing_is_requested(loop, monkeypatch):
    """Фраза распознаётся всегда, а идти с ней может быть некуда."""
    launcher, client = build(monkeypatch)

    answer = loop.run_until_complete(launcher.launch())

    assert not launcher.configured
    assert answer
    assert client.calls == []


def test_network_error_does_not_escape(loop, monkeypatch):
    """Наружу не бросаем: обработчик реплики ждёт готовую фразу."""
    launcher, _ = build(monkeypatch, url=URL, client=FakeClient(error=httpx.ConnectError("нет сети")))

    answer = loop.run_until_complete(launcher.launch())

    assert answer
    assert "Иван Михайлович" not in answer


def test_http_error_does_not_escape(loop, monkeypatch):
    launcher, _ = build(
        monkeypatch, url=URL, client=FakeClient(response=FakeResponse(500, status_error(500)))
    )

    answer = loop.run_until_complete(launcher.launch())

    assert answer
    assert "Иван Михайлович" not in answer


def test_close_without_a_client_is_harmless(loop, monkeypatch):
    launcher, _ = build(monkeypatch, url=URL)

    loop.run_until_complete(launcher.close())


# -------------------------------------------------------------- конфиг ----
def test_settings_come_from_the_config(monkeypatch):
    monkeypatch.delenv(ENV_URL, raising=False)
    launcher = GameLauncher(FakeConfig(url=URL, timeout_s=3, reply="Готово"))

    assert launcher._url == URL
    assert launcher._timeout == 3
    assert launcher._reply == "Готово"
    assert launcher.configured


def test_env_url_wins_over_the_config(monkeypatch):
    """Ссылка с токеном живёт в .env, а не в конфиге, который уезжает в репозиторий."""
    monkeypatch.setenv(ENV_URL, "http://from-env/launch")
    launcher = GameLauncher(FakeConfig(url=URL))

    assert launcher._url == "http://from-env/launch"


def test_empty_env_does_not_erase_the_config(monkeypatch):
    monkeypatch.setenv(ENV_URL, "   ")
    launcher = GameLauncher(FakeConfig(url=URL))

    assert launcher._url == URL


def test_missing_section_falls_back_to_defaults(monkeypatch):
    monkeypatch.delenv(ENV_URL, raising=False)
    launcher = GameLauncher(Empty())

    assert not launcher.configured
    assert launcher._timeout > 0
    assert "Иван Михайлович" in launcher._reply
