"""Анекдоты: разбор ленты, кеш и фолбэк на модель. Без сети."""

from __future__ import annotations

import asyncio

import pytest

from bashmak.jokes import Joke, JokeError, JokeTeller, _parse, _plain_text


def feed(*descriptions: str, guids: list[str] | None = None) -> str:
    """RSS в том виде, в каком его отдаёт лента: текст внутри CDATA."""
    items = []
    for i, text in enumerate(descriptions):
        guid = guids[i] if guids else f"guid-{i}"
        items.append(
            f"<item><title>t</title><link>https://x/{i}</link>"
            f"<description><![CDATA[{text}]]></description>"
            f"<guid>{guid}</guid></item>"
        )
    return f"<rss><channel>{''.join(items)}</channel></rss>"


class FakeConfig:
    """cfg.get('jokes') — единственное, что читает JokeTeller."""

    def __init__(self, **jokes):
        self._section = FakeSection(jokes) if jokes is not None else None

    def get(self, name, default=None):
        return self._section if name == "jokes" else default


class FakeSection:
    def __init__(self, values):
        self._values = values

    def get(self, name, default=None):
        return self._values.get(name, default)


@pytest.fixture()
def loop():
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


def build(monkeypatch, jokes=None, error=None, llm=None, **settings):
    """Рассказчик с подменённой сетью: _fetch() отдаёт что велено."""
    teller = JokeTeller(FakeConfig(**settings), llm)
    calls: list[int] = []

    async def fake_fetch():
        calls.append(1)
        if error is not None:
            raise error
        return list(jokes or [])

    monkeypatch.setattr(teller, "_fetch", fake_fetch)
    return teller, calls


# ------------------------------------------------------------- разбор ----
def test_plain_text_turns_br_into_lines():
    assert _plain_text("Первая<br>Вторая<br/>Третья") == "Первая\nВторая\nТретья"


def test_plain_text_strips_tags_and_entities():
    """Теги синтезатор не пропускает мимо, а читает вслух."""
    assert _plain_text("<b>Он</b> сказал &quot;ага&quot; &amp; ушёл") == 'Он сказал "ага" & ушёл'


def test_plain_text_drops_empty_lines():
    assert _plain_text("Раз<br><br><br>Два   ") == "Раз\nДва"


def test_parse_reads_items():
    jokes = _parse(feed("Первый", "Второй"))

    assert [j.text for j in jokes] == ["Первый", "Второй"]
    assert [j.uid for j in jokes] == ["guid-0", "guid-1"]


def test_parse_skips_items_without_text():
    assert [j.text for j in _parse(feed("", "Есть текст"))] == ["Есть текст"]


def test_parse_falls_back_to_link_when_there_is_no_guid():
    xml = "<rss><channel><item><description>Текст</description>" "<link>https://x/1</link></item></channel></rss>"
    assert _parse(xml)[0].uid == "https://x/1"


def test_parse_rejects_broken_xml():
    with pytest.raises(JokeError):
        _parse("<rss><channel><item>")


def test_parse_of_an_empty_feed_is_empty():
    assert _parse(feed()) == []


# -------------------------------------------------------------- выдача ----
def test_tell_returns_a_joke_from_the_feed(loop, monkeypatch):
    teller, _ = build(monkeypatch, jokes=[Joke("Анекдот", "g1")])

    assert loop.run_until_complete(teller.tell()) == "Анекдот"


def test_feed_is_fetched_once_while_the_cache_is_warm(loop, monkeypatch):
    """Каждая просьба — следующий из пачки, а не поход в сеть."""
    pool = [Joke(f"Анекдот {i}", f"g{i}") for i in range(3)]
    teller, calls = build(monkeypatch, jokes=pool)

    told = [loop.run_until_complete(teller.tell()) for _ in range(3)]

    assert len(calls) == 1
    assert sorted(told) == ["Анекдот 0", "Анекдот 1", "Анекдот 2"]


def test_told_jokes_are_not_repeated_within_a_batch(loop, monkeypatch):
    pool = [Joke(f"Анекдот {i}", f"g{i}") for i in range(5)]
    teller, _ = build(monkeypatch, jokes=pool)

    told = [loop.run_until_complete(teller.tell()) for _ in range(5)]

    assert len(set(told)) == 5


def test_exhausted_batch_sends_the_next_request_back_to_the_feed(loop, monkeypatch):
    """Пачку рассказали целиком — кеш помечается протухшим."""
    teller, calls = build(monkeypatch, jokes=[Joke("Один", "g1")])

    loop.run_until_complete(teller.tell())
    loop.run_until_complete(teller.tell())

    assert len(calls) == 2


def test_stale_batch_is_better_than_silence(loop, monkeypatch):
    """Лента отвалилась, но разобранная пачка уже лежит рядом."""
    teller, _ = build(monkeypatch, jokes=[Joke("Один", "g1"), Joke("Два", "g2")])
    loop.run_until_complete(teller.tell())

    async def dead_fetch():
        raise OSError("сети нет")

    monkeypatch.setattr(teller, "_fetch", dead_fetch)
    teller._fetched_at = 0.0  # кеш протух — пойдёт в сеть и не дойдёт

    assert loop.run_until_complete(teller.tell()) in {"Один", "Два"}


# -------------------------------------------------------------- фолбэк ----
class FakeLlm:
    def __init__(self, reply="", error=None):
        self.reply = reply
        self.error = error
        self.calls: list[dict] = []

    async def complete(self, messages, **kwargs):
        self.calls.append({"messages": messages, **kwargs})
        if self.error is not None:
            raise self.error
        return self.reply


def test_llm_tells_the_joke_when_the_feed_is_down(loop, monkeypatch):
    llm = FakeLlm(reply="Сочинённый анекдот.")
    teller, _ = build(monkeypatch, error=OSError("сети нет"), llm=llm)

    assert loop.run_until_complete(teller.tell()) == "Сочинённый анекдот."
    # Лимит диалога обрезал бы анекдот на середине — у фолбэка свой.
    assert llm.calls[0]["max_tokens"] > 48


def test_llm_reply_is_cleaned_like_any_other(loop, monkeypatch):
    llm = FakeLlm(reply="Анекдот про кота.assistant: а вот ещё")
    teller, _ = build(monkeypatch, error=OSError("сети нет"), llm=llm)

    assert loop.run_until_complete(teller.tell()) == "Анекдот про кота."


def test_feed_wins_over_the_llm(loop, monkeypatch):
    llm = FakeLlm(reply="Сочинённый")
    teller, _ = build(monkeypatch, jokes=[Joke("Из ленты", "g1")], llm=llm)

    assert loop.run_until_complete(teller.tell()) == "Из ленты"
    assert llm.calls == []


def test_both_sources_dead_still_answers(loop, monkeypatch):
    """Наружу не бросаем: обработчик ждёт готовую фразу."""
    llm = FakeLlm(error=RuntimeError("модель легла"))
    teller, _ = build(monkeypatch, error=OSError("сети нет"), llm=llm)

    answer = loop.run_until_complete(teller.tell())

    assert answer
    assert "анекдот" in answer.lower()


def test_without_an_llm_there_is_no_fallback(loop, monkeypatch):
    teller, _ = build(monkeypatch, error=OSError("сети нет"), llm=None)

    assert loop.run_until_complete(teller.tell())


def test_close_without_a_client_is_harmless(loop, monkeypatch):
    teller, _ = build(monkeypatch, jokes=[])
    loop.run_until_complete(teller.close())


# -------------------------------------------------------------- конфиг ----
def test_settings_come_from_the_config():
    teller = JokeTeller(FakeConfig(url="https://x/feed", timeout_s=3, cache_ttl_s=60))

    assert teller._url == "https://x/feed"
    assert teller._timeout == 3
    assert teller._ttl == 60


def test_missing_section_falls_back_to_defaults():
    class Empty:
        def get(self, name, default=None):
            return default

    teller = JokeTeller(Empty())

    assert teller._url.startswith("http")
    assert teller._ttl > 0
