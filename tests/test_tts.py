"""Подготовка текста к синтезу: чистка, нарезка, отсев непроизносимого.

Сам Silero тут не трогаем — torch грузится в процессе-воркере, а эти функции
чистые. Живой круг TTS→VAD→STT проверяет ./scripts/doctor.sh.
"""

from __future__ import annotations

from bashmak.tts import silero_worker
from bashmak.tts.silero_worker import _SPEAKABLE, clean_for_tts, split_sentences


def test_clean_strips_what_cannot_be_spoken():
    cleaned = clean_for_tts("**Вот** тут https://example.com/x 🙂\nи вторая строка")

    assert "*" not in cleaned
    assert "example.com" not in cleaned and "ссылка" in cleaned
    assert "🙂" not in cleaned
    assert "\n" not in cleaned


def test_long_sentence_is_cut_by_words():
    chunks = split_sentences("слово " * 200, max_chars=100)

    assert len(chunks) > 1
    assert all(len(chunk) <= 100 for chunk in chunks)
    # Резать разрешено только по пробелам — обрубка слова слышна.
    assert all(chunk.split() and not chunk.startswith(" ") for chunk in chunks)


def test_unspeakable_chunk_is_detected():
    """Кусок без букв и цифр Silero не переварит — такие пропускаем в stream()."""
    assert not _SPEAKABLE.search("... -- !?")
    assert _SPEAKABLE.search("ага")


class _V5:
    def apply_tts(
        self, text, speaker, sample_rate, put_accent=True, put_yo=True,
        put_stress_homo=True, put_yo_homo=True,
    ):  # pragma: no cover — нужна только сигнатура
        ...


class _V4:
    def apply_tts(
        self, text, speaker, sample_rate, put_accent=True, put_yo=True
    ):  # pragma: no cover — нужна только сигнатура
        ...


_CONFIGURED = {
    "speaker": "aidar",
    "sample_rate": 24000,
    "put_accent": True,
    "put_yo": True,
    "put_stress_homo": True,
    "put_yo_homo": True,
    "threads": 2,
}


def test_v5_keeps_every_put_param(monkeypatch):
    monkeypatch.setattr(silero_worker, "_MODEL", _V5())
    params = dict(_CONFIGURED)

    silero_worker._drop_unsupported(params)

    assert params == _CONFIGURED


def test_older_model_loses_only_homographs(monkeypatch):
    """На модели без омографов лишний kwarg — это TypeError в воркере.

    GuildSession.say ловит любое исключение и только пишет в лог, поэтому такое
    несовпадение выглядит как молчащий бот, а не как ошибка. Отсев на старте —
    единственное, что отличает одно от другого.
    """
    monkeypatch.setattr(silero_worker, "_MODEL", _V4())
    params = dict(_CONFIGURED)

    silero_worker._drop_unsupported(params)

    assert "put_stress_homo" not in params and "put_yo_homo" not in params
    # Ударения, «ё» и всё, что не put_*, трогать не за что.
    assert params["put_accent"] is True and params["put_yo"] is True
    assert params["speaker"] == "aidar" and params["threads"] == 2

    # То, что реально уйдёт в apply_tts, — модель обязана это принять.
    kwargs = {
        "text": "проверка",
        "speaker": params["speaker"],
        "sample_rate": params["sample_rate"],
    }
    kwargs.update({k: v for k, v in params.items() if k.startswith("put_")})
    _V4().apply_tts(**kwargs)
