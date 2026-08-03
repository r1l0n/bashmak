"""Smoke-тест окружения: `./scripts/doctor.sh`.

Проверяет ровно то, что ломается при переносе на новую машину: пакеты,
бинарники, веса, живость llama-server и способность связки Piper→Whisper
вообще выдать звук и прочитать его обратно.

Выход: 0 — всё зелёное, 1 — есть FAIL.
"""

from __future__ import annotations

import argparse
import asyncio
import ctypes.util
import shutil
import sys
import time
from pathlib import Path
from typing import Callable

PASS = "PASS"
FAIL = "FAIL"
WARN = "WARN"

_COLORS = {PASS: "\033[32m", FAIL: "\033[31m", WARN: "\033[33m"}
_OFF = "\033[0m"


class Report:
    def __init__(self) -> None:
        self.rows: list[tuple[str, str, str]] = []
        self.failed = False

    def add(self, status: str, name: str, detail: str = "") -> None:
        self.rows.append((status, name, detail))
        if status == FAIL:
            self.failed = True
        colored = f"{_COLORS[status]}{status}{_OFF}" if sys.stdout.isatty() else status
        print(f"  [{colored}] {name:<28} {detail}")

    def check(self, name: str, fn: Callable[[], str], *, critical: bool = True) -> None:
        try:
            detail = fn()
        except Exception as exc:
            self.add(FAIL if critical else WARN, name, f"{type(exc).__name__}: {exc}")
        else:
            self.add(PASS, name, detail)


def _check_python() -> str:
    major, minor = sys.version_info[:2]
    if not (3, 10) <= (major, minor) <= (3, 12):
        raise RuntimeError(f"{major}.{minor} — нужен Python 3.10–3.12")
    return f"{major}.{minor}"


def _check_ffmpeg() -> str:
    path = shutil.which("ffmpeg")
    if path is None:
        raise RuntimeError("не найден в PATH (apt install ffmpeg)")
    return path


def _check_opus() -> str:
    import discord

    if discord.opus.is_loaded():
        return "загружен"
    library = ctypes.util.find_library("opus")
    if library is None:
        raise RuntimeError("libopus не найдена (apt install libopus0)")
    discord.opus.load_opus(library)
    return library


def _check_imports() -> str:
    import importlib

    modules = [
        "discord",
        "faster_whisper",
        "onnxruntime",
        "soxr",
        "numpy",
        "piper",
        "yt_dlp",
        "rapidfuzz",
        "httpx",
        "yaml",
    ]
    missing = []
    for name in modules:
        try:
            importlib.import_module(name)
        except ImportError:
            missing.append(name)
    if missing:
        raise RuntimeError(f"не импортируются: {', '.join(missing)}")
    return f"{len(modules)} модулей"


def _check_file(path: Path, what: str) -> str:
    if not path.exists():
        raise RuntimeError(f"нет файла {path} — ./scripts/setup.sh")
    size = path.stat().st_size if path.is_file() else sum(
        f.stat().st_size for f in path.rglob("*") if f.is_file()
    )
    return f"{what}, {size / 1024 / 1024:.0f} МБ"


def _check_llama_binary(root: Path) -> str:
    matches = list((root / "vendor").rglob("llama-server"))
    if not matches:
        raise RuntimeError("не найден в vendor/ — ./scripts/setup.sh")
    return str(matches[0])


async def _check_llm(client) -> str:  # noqa: ANN001 — LlmClient
    if not await client.health():
        raise RuntimeError("не отвечает на /health (systemctl status llama-server)")
    started = time.perf_counter()
    reply = await client.complete(
        [{"role": "user", "content": "Ответь одним словом: работает?"}],
        max_tokens=16,
        temperature=0.0,
    )
    if not reply:
        raise RuntimeError("вернул пустой ответ")
    elapsed = time.perf_counter() - started
    return f"{elapsed:.1f} с, ответ {reply[:40]!r}"


async def _check_voice_roundtrip(cfg) -> str:  # noqa: ANN001 — Config
    """Piper синтезирует фразу, Whisper читает её обратно — сквозная проверка."""
    import numpy as np
    import soxr

    from .audio.vad import Segment
    from .stt.whisper_worker import SttPool
    from .tts.piper_worker import TtsPool

    phrase = "Башмак на связи, проверка звука."
    tts = TtsPool(cfg.tts)
    stt = SttPool(cfg.stt)
    try:
        pieces: list[np.ndarray] = []
        rate = 22050
        async for pcm, chunk_rate in tts.stream(phrase):
            pieces.append(np.frombuffer(pcm, dtype="<i2"))
            rate = chunk_rate
        if not pieces:
            raise RuntimeError("Piper не выдал ни одного сэмпла")

        mono = np.concatenate(pieces).astype(np.float32) / 32768.0
        audio = soxr.resample(mono, rate, 16000).astype(np.float32)

        segment = Segment(user_id=0, audio=audio, duration=audio.size / 16000, ended_at=0.0)
        transcript = await stt.transcribe(segment)
    finally:
        tts.close()
        stt.close()

    if transcript is None or not transcript.text.strip():
        raise RuntimeError("Whisper не расслышал синтезированную фразу")
    return f"{audio.size / 16000:.1f} с → {transcript.text.strip()!r}"


async def _check_token(token: str) -> str:
    import httpx

    async with httpx.AsyncClient(timeout=10.0) as client:
        response = await client.get(
            "https://discord.com/api/v10/users/@me",
            headers={"Authorization": f"Bot {token}"},
        )
    if response.status_code == 401:
        raise RuntimeError("Discord не принял токен (401)")
    response.raise_for_status()
    return f"бот {response.json().get('username')}"


async def run(offline: bool) -> int:
    from .config import load_config

    print("\nБашмак — проверка окружения\n")
    report = Report()

    print(" окружение")
    report.check("python", _check_python)
    report.check("ffmpeg", _check_ffmpeg)
    report.check("libopus", _check_opus)
    report.check("python-пакеты", _check_imports)

    print("\n конфигурация")
    try:
        cfg = load_config()
        report.add(PASS, "config.yaml", str(cfg.source))
    except Exception as exc:
        report.add(FAIL, "config.yaml", str(exc))
        print("\nБез конфига дальше проверять нечего.\n")
        return 1

    print("\n модели")
    report.check("VAD", lambda: _check_file(cfg.vad.path("model_path"), "silero"))
    report.check("LLM", lambda: _check_file(cfg.llm.path("model_path"), "gguf"))
    report.check("STT", lambda: _check_file(cfg.stt.path("model_path"), "faster-whisper"))
    report.check("TTS", lambda: _check_file(cfg.tts.path("voice_path"), "piper"))
    report.check("llama-server", lambda: _check_llama_binary(cfg.root))

    print("\n рантайм")
    if offline:
        report.add(WARN, "llama-server /health", "пропущено (--offline)")
        report.add(WARN, "TTS→STT", "пропущено (--offline)")
        report.add(WARN, "DISCORD_TOKEN", "пропущено (--offline)")
        print()
        return 1 if report.failed else 0

    from .llm.client import LlmClient

    client = LlmClient(cfg.llm)
    try:
        detail = await _check_llm(client)
        report.add(PASS, "llama-server", detail)
    except Exception as exc:
        report.add(FAIL, "llama-server", f"{type(exc).__name__}: {exc}")
    finally:
        await client.close()

    try:
        report.add(PASS, "TTS→STT", await _check_voice_roundtrip(cfg))
    except Exception as exc:
        report.add(FAIL, "TTS→STT", f"{type(exc).__name__}: {exc}")

    try:
        report.add(PASS, "DISCORD_TOKEN", await _check_token(cfg.discord_token))
    except Exception as exc:
        report.add(FAIL, "DISCORD_TOKEN", f"{type(exc).__name__}: {exc}")

    total = len(report.rows)
    failed = sum(1 for status, _, _ in report.rows if status == FAIL)
    print(f"\n Итог: {total - failed}/{total} проверок пройдено\n")
    return 1 if report.failed else 0


def main() -> None:
    parser = argparse.ArgumentParser(description="Проверка окружения Башмака")
    parser.add_argument(
        "--offline",
        action="store_true",
        help="не трогать сеть и не поднимать модели (только файлы и пакеты)",
    )
    args = parser.parse_args()
    sys.exit(asyncio.run(run(args.offline)))


if __name__ == "__main__":
    main()
