"""Выбор движка STT: ошибки должны быть внятными, а шаблон — рабочим.

Повод тот же, что у test_config.py. Настройки распознавания переехали из общей
секции ``stt:`` в подсекции ``stt.gigaam`` и ``stt.whisper``, и у старых
конфигов ключи оказываются не там, где их ищет фабрика. Отказ при этом самый
неудобный из возможных: движок либо не поднимается вовсе, либо поднимается на
значениях по умолчанию, и бот молча начинает хуже слышать.

Сами движки здесь не создаются — для этого нужны веса на диске, а тест должен
проходить на машине без моделей. Проверяется разбор конфига до загрузки весов.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from bashmak.config import Section
from bashmak.stt import Transcript, create_stt_pool

ROOT = Path("/opt/bashmak")


def test_unknown_engine_names_the_valid_ones():
    cfg = Section({"engine": "wisper", "gigaam": {}, "whisper": {}}, ROOT)

    with pytest.raises(ValueError) as excinfo:
        create_stt_pool(cfg)

    message = str(excinfo.value)
    assert "wisper" in message, "в ошибке должно быть то, что написано в конфиге"
    assert "gigaam" in message and "whisper" in message


def test_missing_engine_section_says_which_one():
    """Ровно случай старого config.yaml: engine есть, подсекции под него нет."""
    cfg = Section({"engine": "gigaam"}, ROOT)

    with pytest.raises(ValueError) as excinfo:
        create_stt_pool(cfg)

    assert "stt.gigaam" in str(excinfo.value)


def test_engine_name_is_case_and_space_tolerant():
    cfg = Section({"engine": " GigaAM ", "gigaam": {}}, ROOT)

    # Дальше фабрика полезет за весами, и это уже не её ошибка. Важно, что до
    # выбора движка дело дошло: ValueError про неизвестный движок не поднялся.
    with pytest.raises(Exception) as excinfo:
        create_stt_pool(cfg)

    assert "неизвестный stt.engine" not in str(excinfo.value)


def test_template_default_engine_has_its_section():
    """Шаблон должен быть согласован сам с собой: engine и подсекция под него."""
    import yaml

    from bashmak.config import EXAMPLE

    with EXAMPLE.open(encoding="utf-8") as fh:
        stt = yaml.safe_load(fh)["stt"]

    assert stt["engine"] in stt, f"в шаблоне нет секции stt.{stt['engine']}"
    assert stt[stt["engine"]]["model_path"], "у движка по умолчанию не задан model_path"


def test_transcript_keeps_end_of_phrase_time():
    """ended_at переносится как есть — по нему очередь LLM держит порядок."""
    transcript = Transcript(
        user_id=1, text="привет", duration=1.0, avg_logprob=-0.2, ended_at=1234.5
    )

    assert transcript.ended_at == 1234.5
