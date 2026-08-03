"""TTS: Piper в пуле процессов.

Синтез блокирующий и заметно грузит CPU, поэтому он вынесен из лупа — иначе
бот переставал бы принимать голос, пока говорит сам.

Длинный ответ режется на предложения и отдаётся наружу по кускам: первое
предложение начинает звучать, пока синтезируется второе. На CPU это разница
между «отвечает сразу» и «думает три секунды и выдаёт всё разом».

Ответ модели перед синтезом чистится: Piper озвучивает звёздочки, решётки и
эмодзи буквально.
"""

from __future__ import annotations

import asyncio
import logging
import multiprocessing
import re
from concurrent.futures import ProcessPoolExecutor
from typing import AsyncIterator

from ..utils.logging import stage

log = logging.getLogger(__name__)

_VOICE = None
_SYNTH_KWARGS: dict[str, float] = {}

# Разметка, ссылки и эмодзи — всё, что нельзя произносить.
_MARKDOWN = re.compile(r"[*_`#>~|\[\]]+")
_URL = re.compile(r"https?://\S+")
_EMOJI = re.compile(
    "[" "\U0001f300-\U0001faff" "\U00002600-\U000027bf" "\U0001f1e6-\U0001f1ff" "]+",
    flags=re.UNICODE,
)
_SENTENCE = re.compile(r"(?<=[.!?…])\s+")
_SPACES = re.compile(r"\s+")


def clean_for_tts(text: str) -> str:
    cleaned = _URL.sub(" ссылка ", text)
    cleaned = _EMOJI.sub(" ", cleaned)
    cleaned = _MARKDOWN.sub(" ", cleaned)
    cleaned = cleaned.replace("\n", ". ")
    return _SPACES.sub(" ", cleaned).strip()


def split_sentences(text: str, max_chars: int = 220) -> list[str]:
    """Разбить на куски, удобные для потокового синтеза."""
    chunks: list[str] = []
    buffer = ""
    for sentence in _SENTENCE.split(text):
        sentence = sentence.strip()
        if not sentence:
            continue
        if buffer and len(buffer) + len(sentence) + 1 > max_chars:
            chunks.append(buffer)
            buffer = sentence
        else:
            buffer = f"{buffer} {sentence}".strip()
    if buffer:
        chunks.append(buffer)
    return chunks


def _init_worker(voice_path: str, params: dict[str, float]) -> None:
    global _VOICE, _SYNTH_KWARGS
    from piper import PiperVoice

    _VOICE = PiperVoice.load(voice_path)
    _SYNTH_KWARGS = params


def _synthesis_config(params: dict[str, float]):
    """Собрать SynthesisConfig, оставив только поля, которые знает эта версия Piper.

    Между piper 1.2 (rhasspy) и piper1-gpl 1.4 поля переименовывались
    (noise_w → noise_w_scale), а окружение на сервере обновляется отдельно от
    кода — проще подстроиться, чем пинить версию намертво.
    """
    try:
        from dataclasses import fields

        from piper import SynthesisConfig
    except ImportError:
        return None

    allowed = {f.name for f in fields(SynthesisConfig)}
    aliases = {"noise_w": "noise_w_scale"}
    kwargs = {}
    for key, value in params.items():
        name = key if key in allowed else aliases.get(key, key)
        if name in allowed:
            kwargs[name] = value
    return SynthesisConfig(**kwargs)


def _synthesize(text: str) -> tuple[bytes, int]:
    """Выполняется в процессе-воркере. Возвращает (моно int16 PCM, частота)."""
    if _VOICE is None:  # pragma: no cover
        raise RuntimeError("голос Piper не инициализирован в воркере")

    parts: list[bytes] = []
    sample_rate = getattr(getattr(_VOICE, "config", None), "sample_rate", 22050)

    synthesize = getattr(_VOICE, "synthesize", None)
    if synthesize is not None:
        config = _synthesis_config(_SYNTH_KWARGS)
        stream = synthesize(text, config) if config is not None else synthesize(text)
        for chunk in stream:
            audio = getattr(chunk, "audio_int16_bytes", None)
            if audio is None:  # старый API отдавал сырые байты
                parts.append(bytes(chunk))
            else:
                parts.append(audio)
                sample_rate = getattr(chunk, "sample_rate", sample_rate)
    else:  # piper < 1.3
        parts.extend(_VOICE.synthesize_stream_raw(text, **_SYNTH_KWARGS))

    return b"".join(parts), sample_rate


class TtsPool:
    def __init__(self, cfg) -> None:  # noqa: ANN001 — bashmak.config.Section
        voice_path = cfg.path("voice_path")
        if not voice_path.exists():
            raise FileNotFoundError(
                f"нет голоса Piper: {voice_path}. Запустите ./scripts/setup.sh"
            )

        params = {
            "length_scale": float(cfg.get("length_scale", 1.0)),
            "noise_scale": float(cfg.get("noise_scale", 0.667)),
            "noise_w": float(cfg.get("noise_w", 0.8)),
        }
        workers = int(cfg.get("workers", 1))
        self._pool = ProcessPoolExecutor(
            max_workers=workers,
            mp_context=multiprocessing.get_context("spawn"),
            initializer=_init_worker,
            initargs=(str(voice_path), params),
        )
        log.info("TTS: %s, воркеров %d", voice_path.name, workers)

    async def stream(self, text: str) -> AsyncIterator[tuple[bytes, int]]:
        """Синтезировать текст кусками: (моно int16 PCM, частота)."""
        cleaned = clean_for_tts(text)
        if not cleaned:
            return

        loop = asyncio.get_running_loop()
        for index, chunk in enumerate(split_sentences(cleaned)):
            with stage(log, f"tts[{index}]", chars=len(chunk)):
                pcm, rate = await loop.run_in_executor(self._pool, _synthesize, chunk)
            if pcm:
                yield pcm, rate

    def close(self) -> None:
        self._pool.shutdown(wait=False, cancel_futures=True)
