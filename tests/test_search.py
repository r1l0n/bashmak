"""Случайный выбор трека: чистые части, без сети и без yt-dlp."""

from __future__ import annotations

import pytest

from bashmak.music.search import _choose_url, _source_query


def entry(**overrides):
    """Плоская запись выдачи: годная, пока её явно не испортили."""
    data = {"id": "abc", "url": "https://x/abc", "duration": 200}
    data.update(overrides)
    return data


def test_source_query_wraps_plain_text():
    assert _source_query("русский рок", 20) == "ytsearch20:русский рок"


@pytest.mark.parametrize(
    "url",
    [
        "https://www.youtube.com/playlist?list=PL123",
        "http://youtu.be/abc",
    ],
)
def test_source_query_keeps_links(url):
    assert _source_query(url, 20) == url


def test_choose_url_takes_the_only_good_entry():
    entries = [
        entry(url="https://x/live", is_live=True),
        # Часовой сборник: радио встало бы на нём до вечера.
        entry(url="https://x/long", duration=36000),
        entry(url="https://x/ok"),
    ]
    assert _choose_url(entries) == "https://x/ok"


def test_choose_url_skips_live_status_flag():
    entries = [entry(url="https://x/stream", live_status="is_live"), entry(url="https://x/ok")]
    assert _choose_url(entries) == "https://x/ok"


def test_choose_url_skips_recent():
    entries = [entry(url="https://x/a"), entry(url="https://x/b")]
    assert _choose_url(entries, {"https://x/a"}) == "https://x/b"


def test_choose_url_keeps_entries_without_duration():
    """Длительность в плоской выдаче есть не всегда — это не повод отсеивать."""
    assert _choose_url([entry(url="https://x/ok", duration=None)]) == "https://x/ok"


def test_choose_url_builds_link_from_id():
    """Часть экстракторов отдаёт только id, готовой ссылки в записи нет."""
    assert _choose_url([{"id": "xyz"}]) == "https://www.youtube.com/watch?v=xyz"


@pytest.mark.parametrize(
    "entries",
    [
        [],
        [entry(is_live=True)],
        [{}],
        # Ни ссылки, ни id — брать нечего.
        [{"title": "без ссылки"}],
    ],
)
def test_choose_url_returns_empty_when_nothing_fits(entries):
    assert _choose_url(entries) == ""
