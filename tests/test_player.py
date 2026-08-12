"""Плеер: автовыбор трека и радио.

Сеть и ffmpeg подменены заглушками — проверяется только логика самого плеера:
куда уходит пустой запрос, когда встаёт радио и когда оно обязано молчать.
"""

from __future__ import annotations

import asyncio

import pytest

from bashmak.music import player as player_module
from bashmak.music.player import MusicPlayer
from bashmak.music.search import SearchError, Track


class FakeSource:
    """Микшер в объёме, который трогает плеер."""

    def __init__(self):
        self.music = None
        self.volume = 0.5
        self.music_paused = False

    def set_music(self, source):
        self.music = source

    def pause_music(self):
        self.music_paused = True

    def resume_music(self):
        self.music_paused = False

    def set_volume(self, value):
        self.volume = max(0.0, min(1.5, float(value)))
        return self.volume


class FakeArbiter:
    def __init__(self):
        self.source = FakeSource()
        self.on_music_end = None

    def bind_music_end(self, callback):
        self.on_music_end = callback


class FakeConfig:
    """cfg.music — плеер читает секцию только через get()."""

    def __init__(self, **music):
        defaults = {
            "max_queue": 20,
            "search_timeout_s": 1,
            "random_sources": ["рок"],
            "random_pool": 5,
            "random_min_views": 1_000_000,
            "autoplay": True,
        }
        defaults.update(music)
        self.music = defaults


def track(title="песня", url="https://x/1") -> Track:
    return Track(title=title, stream_url="https://stream/1", page_url=url, duration=100)


@pytest.fixture()
def loop():
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest.fixture(autouse=True)
def no_ffmpeg(monkeypatch):
    """_start() иначе полезет поднимать настоящий процесс ffmpeg."""
    monkeypatch.setattr(
        player_module.discord, "FFmpegPCMAudio", lambda *args, **kwargs: object()
    )


def build(**music):
    player = MusicPlayer(FakeConfig(**music), FakeArbiter())
    return player, player._arbiter


def in_loop(loop, method, *args):
    """Дёрнуть синхронный метод плеера изнутри лупа.

    Радио заводится через ``asyncio.create_task``, а он требует работающего
    лупа — снаружи метод молча решил бы, что бот выключается.
    """

    async def call():
        return method(*args)

    return loop.run_until_complete(call())


def drain_radio(loop, player):
    """Дождаться заведённого радио-поиска."""
    assert player._radio_task is not None, "радио не встало"
    loop.run_until_complete(asyncio.wait_for(player._radio_task, 1.0))


def stub_random(monkeypatch, tracks=None, error: Exception | None = None):
    """Подменить случайный выбор и запомнить, с чем его звали."""
    calls: list[dict] = []
    queue = list(tracks or [])

    async def fake_search_random(sources, pool, timeout, exclude=(), min_views=0):
        calls.append(
            {
                "sources": list(sources),
                "pool": pool,
                "exclude": set(exclude),
                "min_views": min_views,
            }
        )
        if error is not None:
            raise error
        return queue.pop(0) if queue else track()

    monkeypatch.setattr(player_module, "search_random", fake_search_random)
    return calls


def test_empty_query_picks_a_track_itself(loop, monkeypatch):
    player, _ = build()
    calls = stub_random(monkeypatch, [track("Кино — Группа крови")])

    answer = loop.run_until_complete(player.play("", "балбес"))

    assert answer == "Сам выбрал: Кино — Группа крови"
    assert player.current is not None
    assert calls[0]["sources"] == ["рок"]
    # Порог популярности доезжает из конфига до поиска, а не теряется по пути.
    assert calls[0]["min_views"] == 1_000_000


def test_empty_query_without_sources_still_asks(loop, monkeypatch):
    player, _ = build(random_sources=[])
    stub_random(monkeypatch)

    answer = loop.run_until_complete(player.play(""))

    assert answer == "А что включить-то? Скажи название."
    assert player.current is None


def test_failed_random_search_does_not_break_the_command(loop, monkeypatch):
    player, _ = build()
    stub_random(monkeypatch, error=SearchError("выдача пуста"))

    answer = loop.run_until_complete(player.play(""))

    assert answer == "Не нашёл, чего бы поставить. Скажи название."
    assert player.current is None


def test_named_track_keeps_its_phrase(loop, monkeypatch):
    player, _ = build()

    async def fake_search(query, timeout):
        return track("Наутилус — Скованные")

    monkeypatch.setattr(player_module, "search", fake_search)

    assert loop.run_until_complete(player.play("наутилус")) == "Включаю: Наутилус — Скованные"


def test_recent_tracks_are_excluded_from_the_next_pick(loop, monkeypatch):
    """Радио не должно крутить то, что только что играло."""
    player, _ = build()
    calls = stub_random(monkeypatch, [track("первый", "https://x/1"), track("второй", "https://x/2")])

    loop.run_until_complete(player.play(""))
    in_loop(loop, player._on_track_end, player._source)
    drain_radio(loop, player)

    assert player.current is not None
    assert player.current.title == "второй"
    assert calls[1]["exclude"] == {"https://x/1"}


def test_radio_picks_up_when_the_queue_runs_dry(loop, monkeypatch):
    player, _ = build()
    stub_random(monkeypatch, [track("первый", "https://x/1"), track("второй", "https://x/2")])

    loop.run_until_complete(player.play(""))
    in_loop(loop, player._on_track_end, player._source)

    drain_radio(loop, player)
    assert player.current.title == "второй"


def test_radio_stays_off_when_autoplay_is_disabled(loop, monkeypatch):
    player, _ = build(autoplay=False)
    stub_random(monkeypatch, [track("первый"), track("второй")])

    loop.run_until_complete(player.play(""))
    in_loop(loop, player._on_track_end, player._source)

    assert player._radio_task is None
    assert player.current is None
    assert player.radio_on is False


def test_stop_disarms_the_radio(loop, monkeypatch):
    player, _ = build()
    stub_random(monkeypatch, [track("первый"), track("второй")])

    loop.run_until_complete(player.play(""))
    assert player.radio_on is True

    assert in_loop(loop, player.stop) == "Выключил музыку."
    assert player.radio_on is False

    # Конец трека, доехавший после команды, радио не заводит.
    in_loop(loop, player._on_track_end, object())
    assert player._radio_task is None


def test_radio_does_not_start_when_something_else_already_plays(loop, monkeypatch):
    """Пока шёл поиск, человек включил своё — наш трек уже не нужен."""
    player, _ = build()
    stub_random(monkeypatch, [track("радийный", "https://x/radio")])
    player._radio_armed = True
    player._current = track("человеческий", "https://x/human")

    loop.run_until_complete(player._radio_next())

    assert player.current.title == "человеческий"


def test_radio_gives_up_after_a_failed_search(loop, monkeypatch):
    player, _ = build()
    stub_random(monkeypatch, error=SearchError("сеть отвалилась"))
    player._radio_armed = True

    loop.run_until_complete(player._radio_next())

    assert player.radio_on is False


def test_skip_on_an_empty_queue_hands_over_to_the_radio(loop, monkeypatch):
    player, _ = build()
    stub_random(monkeypatch, [track("первый"), track("второй")])

    loop.run_until_complete(player.play(""))
    answer = in_loop(loop, player.skip)

    assert answer == "Пропустил первый. Ищу следующий."
    drain_radio(loop, player)
    assert player.current.title == "второй"


def test_skip_reports_an_empty_queue_without_the_radio(loop, monkeypatch):
    player, _ = build(autoplay=False)
    stub_random(monkeypatch, [track("первый")])

    loop.run_until_complete(player.play(""))

    assert in_loop(loop, player.skip) == "Пропустил первый. Очередь пуста."
