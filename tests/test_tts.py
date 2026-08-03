"""Подготовка текста к синтезу: чистка, нарезка, отсев непроизносимого.

Сам Silero тут не трогаем — torch грузится в процессе-воркере, а эти функции
чистые. Живой круг TTS→VAD→STT проверяет ./scripts/doctor.sh.
"""

from __future__ import annotations

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
