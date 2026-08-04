"""Нечёткий поиск имени бота."""

from __future__ import annotations

import pytest

from bashmak.wakeword.filter import WakeWordFilter, normalize


class FakeSection:
    """Мини-заглушка bashmak.config.Section."""

    def __init__(self, **data):
        self._data = data

    def get(self, name, default=None):
        return self._data.get(name, default)


@pytest.fixture()
def wake():
    return WakeWordFilter(
        FakeSection(
            variants=["башмак", "бошмак", "башмачок", "бащмак", "башмачек"],
            threshold=80,
            strip_prefix=True,
        )
    )


@pytest.mark.parametrize(
    "text, payload",
    [
        ("Башмак, расскажи анекдот", "расскажи анекдот"),
        ("башмак включи музыку", "включи музыку"),
        # Whisper регулярно слышит имя со склонением или с ошибкой.
        ("бошмак, привет", "привет"),
        ("Башмака попроси включить трек", "попроси включить трек"),
        # «эй» остаётся: вырезаем имя, а не всё, что стоит перед ним.
        ("эй башмачок сделай потише", "эй, сделай потише"),
        # Имя в конце — обращение после запроса.
        ("Привет, Башмак", "Привет"),
        ("Включи дору, Башмак", "Включи дору"),
        ("Включи Кино, башмачок", "Включи Кино"),
    ],
)
def test_wake_word_found(wake, text, payload):
    match = wake.match(text)
    assert match is not None
    assert match.payload == payload


def test_name_in_the_middle_keeps_both_sides(wake):
    """Раньше всё до имени выбрасывалось, и запрос терялся целиком."""
    match = wake.match("Включи дору Башмак пожалуйста")

    assert match is not None
    assert match.payload == "Включи дору, пожалуйста"


def test_every_occurrence_of_the_name_is_cut_out(wake):
    """Второе имя не должно уехать в модель как часть вопроса."""
    match = wake.match("Башмак, привет, Башмак")

    assert match is not None
    assert match.payload == "привет"


@pytest.mark.parametrize(
    "text",
    [
        "привет всем в канале",
        "бабушка приехала",
        "да ну нет",
        "",
        "машина сломалась",
    ],
)
def test_wake_word_absent(wake, text):
    assert wake.match(text) is None


def test_bare_name_is_still_an_address(wake):
    """«Башмак!» без продолжения — тоже обращение, просто без запроса."""
    match = wake.match("Башмак!")
    assert match is not None
    assert match.payload == ""


def test_prefix_kept_when_configured():
    wake = WakeWordFilter(FakeSection(variants=["башмак"], threshold=80, strip_prefix=False))
    match = wake.match("Башмак, что там по погоде?")
    assert match is not None
    assert match.payload == "Башмак, что там по погоде?"


def test_payload_keeps_original_case_and_punctuation():
    """Нормализация — только для сравнения: дальше по пайплайну уходит живой текст."""
    wake = WakeWordFilter(FakeSection(variants=["башмак"], threshold=80, strip_prefix=True))
    match = wake.match("Башмак, расскажи анекдот про кота!")
    assert match is not None
    assert match.payload == "расскажи анекдот про кота!"


def test_normalize_strips_punctuation_and_yo():
    assert normalize("Ёжик, привет!!!") == "ежик привет"
