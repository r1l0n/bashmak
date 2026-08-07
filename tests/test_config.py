"""Пути из конфига и внятность ошибки, когда ключа нет.

Повод для теста живой: после переименования tts.voice_path → tts.model_path
доктор отчитался «KeyError: 'model_path'» — ни секции, ни подсказки, что
делать.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from bashmak.config import Section, merge_defaults

ROOT = Path("/opt/bashmak")


def test_relative_path_is_resolved_from_project_root():
    section = Section({"model_path": "models/tts/v4_ru.pt"}, ROOT)
    assert section.path("model_path") == ROOT / "models/tts/v4_ru.pt"


def test_absolute_path_is_left_alone():
    section = Section({"model_path": "/srv/models/v4_ru.pt"}, ROOT)
    assert section.path("model_path") == Path("/srv/models/v4_ru.pt")


def test_missing_key_names_what_is_actually_there():
    section = Section({"voice_path": "models/tts/ru_RU-irina-medium.onnx"}, ROOT)

    with pytest.raises(AttributeError) as excinfo:
        section.path("model_path")

    message = str(excinfo.value)
    assert "model_path" in message
    assert "voice_path" in message, "в ошибке должны быть перечислены реальные ключи"


# ------------------------------------------ шаблон как значения по умолчанию --
def test_new_template_key_reaches_an_old_config():
    """setup.sh существующий config.yaml не трогает — ключ должен доехать сам."""
    merged = merge_defaults(
        {"music": {"volume": 0.5, "duck_hold_ms": 400}}, {"music": {"volume": 0.8}}
    )

    assert merged["music"] == {"volume": 0.8, "duck_hold_ms": 400}


def test_user_value_wins_over_the_template():
    merged = merge_defaults({"llm": {"threads": 7}}, {"llm": {"threads": 4}})
    assert merged["llm"]["threads"] == 4


def test_list_is_replaced_not_appended():
    """guild_ids в шаблоне пуст: пользовательский список должен его вытеснить."""
    merged = merge_defaults({"discord": {"guild_ids": []}}, {"discord": {"guild_ids": [7]}})
    assert merged["discord"]["guild_ids"] == [7]


def test_unknown_key_is_reported_but_does_not_break_startup(caplog):
    """Раньше опечатка молча не читалась, и человек искал, почему нет эффекта."""
    with caplog.at_level("WARNING"):
        merged = merge_defaults({"vad": {"threshold": 0.5}}, {"vad": {"treshold": 0.9}})

    assert "vad.treshold" in caplog.text
    assert merged["vad"]["threshold"] == 0.5, "настоящий ключ обязан остаться дефолтным"
