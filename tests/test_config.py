"""Пути из конфига и внятность ошибки, когда ключа нет.

Повод для теста живой: после переименования tts.voice_path → tts.model_path
доктор отчитался «KeyError: 'model_path'» — ни секции, ни подсказки, что
делать.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from bashmak.config import Section

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
