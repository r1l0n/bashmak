"""TTS: Silero v5 (ru) в пуле процессов.

Выбран вместо Piper: у Piper для русского все голоса «medium» и звучат
синтетично, Silero ставит ударения, произносит «ё», снимает омографы
(за́мок / замо́к) и держит вопросительную интонацию. Цена — torch в
зависимостях и загрузка модели в каждый процесс-воркер, поэтому по умолчанию
``workers: 1``.

Синтез блокирующий и нагружает CPU, поэтому вынесен из лупа: иначе бот
переставал бы принимать голос, пока говорит сам.

Длинный ответ режется на предложения и отдаётся наружу кусками: первое
предложение начинает звучать, пока синтезируется второе. На CPU это заметно
сокращает время до начала ответа.

Текст перед синтезом чистится: разметку и эмодзи синтезатор озвучивает
буквально.
"""

from __future__ import annotations

import asyncio
import inspect
import logging
import multiprocessing
import re
from concurrent.futures import ProcessPoolExecutor
from typing import AsyncIterator

from ..utils.logging import stage

log = logging.getLogger(__name__)

_MODEL = None
_PARAMS: dict = {}

# Разметка, ссылки и эмодзи — всё, что нельзя произносить.
_MARKDOWN = re.compile(r"[*_`#>~|\[\]]+")
_URL = re.compile(r"https?://\S+")
_EMOJI = re.compile(
    "[" "\U0001f300-\U0001faff" "\U00002600-\U000027bf" "\U0001f1e6-\U0001f1ff" "]+",
    flags=re.UNICODE,
)
_SENTENCE = re.compile(r"(?<=[.!?…])\s+")
_SPACES = re.compile(r"\s+")
#: Кусок без единой буквы или цифры Silero произнести не может и падает
#: с «No text left after cleaning» — такие пропускаем.
_SPEAKABLE = re.compile(r"[^\W_]", flags=re.UNICODE)


def clean_for_tts(text: str) -> str:
    cleaned = _URL.sub(" ссылка ", text)
    cleaned = _EMOJI.sub(" ", cleaned)
    cleaned = _MARKDOWN.sub(" ", cleaned)
    cleaned = cleaned.replace("\n", ". ")
    return _SPACES.sub(" ", cleaned).strip()


def _split_long(sentence: str, max_chars: int) -> list[str]:
    """Дорезать предложение длиннее max_chars — по словам, а не по буквам.

    Иначе длинное предложение без точек уходит в синтез целиком, и потоковый
    синтез теряет смысл: ничего не звучит, пока не досчитается весь кусок.
    """
    if len(sentence) <= max_chars:
        return [sentence]

    parts: list[str] = []
    buffer = ""
    for word in sentence.split(" "):
        if buffer and len(buffer) + len(word) + 1 > max_chars:
            parts.append(buffer)
            buffer = word
        else:
            buffer = f"{buffer} {word}".strip()
    if buffer:
        parts.append(buffer)
    return parts


def split_sentences(text: str, max_chars: int = 220) -> list[str]:
    """Разбить на куски, удобные для потокового синтеза."""
    chunks: list[str] = []
    buffer = ""
    for sentence in _SENTENCE.split(text):
        for piece in _split_long(sentence.strip(), max_chars):
            if not piece:
                continue
            if buffer and len(buffer) + len(piece) + 1 > max_chars:
                chunks.append(buffer)
                buffer = piece
            else:
                buffer = f"{buffer} {piece}".strip()
    if buffer:
        chunks.append(buffer)
    return chunks


def _accepted_kwargs(func) -> set[str] | None:  # noqa: ANN001 — метод из torch.package
    """Имена именованных аргументов func. None — если проверить нечем."""
    try:
        signature = inspect.signature(func)
    except (TypeError, ValueError):  # pragma: no cover — сигнатура нечитаема
        return None
    kinds = (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
    if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in signature.parameters.values()):
        return None  # **kwargs принимает что угодно, отсеивать нечего
    return {name for name, p in signature.parameters.items() if p.kind in kinds}


def _drop_unsupported(params: dict) -> None:
    """Выбросить ключи put_*, которых нет в apply_tts этой модели.

    Набор put_* меняется от версии к версии: омографы (put_stress_homo,
    put_yo_homo) появились только в v5. Лишний kwarg — это TypeError в воркере,
    а GuildSession.say ловит любое исключение и лишь пишет в лог, так что
    несовпадение конфига и версии модели выглядит как молчащий бот. Дешевле
    сверить один раз на старте — как resample.py щупает soxr.ResampleStream.
    """
    allowed = _accepted_kwargs(_MODEL.apply_tts)
    if allowed is None:
        return
    dropped = [key for key in params if key.startswith("put_") and key not in allowed]
    for key in dropped:
        del params[key]
    if dropped:
        log.warning("модель не принимает %s — параметры пропущены", ", ".join(sorted(dropped)))


def _init_worker(model_path: str, params: dict) -> None:
    global _MODEL, _PARAMS
    import torch

    # Без этого torch забирает все ядра и конкурирует с Whisper и llama.
    torch.set_num_threads(int(params.get("threads", 2)))
    # Модель — torch.package, а не обычный state_dict: внутри архива лежит и
    # код, и веса, поэтому грузится importer'ом, а не torch.load.
    _MODEL = torch.package.PackageImporter(model_path).load_pickle("tts_models", "model")
    _MODEL.to(torch.device("cpu"))
    _drop_unsupported(params)
    _PARAMS = params
    # Первый вызов torch тратит секунды на прогрев (аллокаторы, ядра). Эти
    # секунды должны уйти в старт бота, а не в первый ответ пользователю.
    _synthesize("раз")


def _synthesize(text: str) -> tuple[bytes, int]:
    """Выполняется в процессе-воркере. Возвращает (моно int16 PCM, частота)."""
    if _MODEL is None:  # pragma: no cover
        raise RuntimeError("модель Silero не инициализирована в воркере")

    import numpy as np

    rate = int(_PARAMS.get("sample_rate", 24000))
    kwargs = {
        "text": text,
        "speaker": _PARAMS.get("speaker", "aidar"),
        "sample_rate": rate,
    }
    # Словарём, а не списком именованных аргументов: _drop_unsupported уже вычистил
    # отсюда всё, чего эта версия модели не понимает.
    kwargs.update({key: bool(value) for key, value in _PARAMS.items() if key.startswith("put_")})
    audio = _MODEL.apply_tts(**kwargs)
    samples = np.asarray(audio.numpy(), dtype=np.float32)
    pcm = np.clip(samples * 32767.0, -32768, 32767).astype("<i2")
    return pcm.tobytes(), rate


class TtsPool:
    def __init__(self, cfg) -> None:  # noqa: ANN001 — bashmak.config.Section
        model_path = cfg.path("model_path")
        if not model_path.exists():
            raise FileNotFoundError(
                f"нет модели Silero: {model_path}. Запустите ./scripts/setup.sh"
            )

        params = {
            "speaker": str(cfg.get("speaker", "aidar")),
            "sample_rate": int(cfg.get("sample_rate", 24000)),
            "put_accent": bool(cfg.get("put_accent", True)),
            "put_yo": bool(cfg.get("put_yo", True)),
            "put_stress_homo": bool(cfg.get("put_stress_homo", True)),
            "put_yo_homo": bool(cfg.get("put_yo_homo", True)),
            "threads": int(cfg.get("threads", 2)),
        }
        workers = int(cfg.get("workers", 1))
        self._pool = ProcessPoolExecutor(
            max_workers=workers,
            mp_context=multiprocessing.get_context("spawn"),
            initializer=_init_worker,
            initargs=(str(model_path), params),
        )
        log.info("TTS: %s, голос %s, воркеров %d", model_path.name, params["speaker"], workers)

    async def stream(self, text: str) -> AsyncIterator[tuple[bytes, int]]:
        """Синтезировать текст кусками: (моно int16 PCM, частота)."""
        cleaned = clean_for_tts(text)
        if not cleaned:
            return

        loop = asyncio.get_running_loop()
        for index, chunk in enumerate(split_sentences(cleaned)):
            if not _SPEAKABLE.search(chunk):
                continue
            with stage(log, f"tts[{index}]", chars=len(chunk)):
                pcm, rate = await loop.run_in_executor(self._pool, _synthesize, chunk)
            if pcm:
                yield pcm, rate

    def close(self) -> None:
        self._pool.shutdown(wait=False, cancel_futures=True)
