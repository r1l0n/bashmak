"""Smoke-тест окружения: `./scripts/doctor.sh`.

Проверяет то, что ломается при переносе на новую машину: пакеты, бинарники,
веса, живость llama-server и способность связки TTS→Whisper выдать звук и
прочитать его обратно.

Выход: 0 — всё зелёное, 1 — есть FAIL.
"""

from __future__ import annotations

import argparse
import asyncio
import ctypes.util
import os
import shutil
import socket
import subprocess
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
        # Приём голоса и E2EE. Проверяется здесь, а не при первом /start: если
        # в venv остался py-cord, discord.ext.voice_recv не импортируется.
        "discord.ext.voice_recv",
        "davey",
        "faster_whisper",
        "onnxruntime",
        "soxr",
        "numpy",
        "torch",
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
    """TTS синтезирует фразу, VAD признаёт её речью, STT читает обратно.

    VAD включён в проверку потому, что при поломке он не падает, а молчит.
    Сломанный VAD выглядит как полностью рабочий бот, который никогда никого не
    слышит; обнаружить это можно только прогнав через него заведомую речь.
    """
    import numpy as np
    import soxr

    from .audio.vad import WINDOW_SAMPLES_16K, Segment, SileroVad
    from .stt import create_stt_pool
    from .tts.silero_worker import TtsPool

    phrase = "Башмак на связи, проверка звука."
    tts = TtsPool(cfg.tts)
    stt = create_stt_pool(cfg.stt)
    try:
        pieces: list[np.ndarray] = []
        rate = 48000
        async for pcm, chunk_rate in tts.stream(phrase):
            pieces.append(np.frombuffer(pcm, dtype="<i2"))
            rate = chunk_rate
        if not pieces:
            raise RuntimeError("синтезатор не выдал ни одного сэмпла")

        mono = np.concatenate(pieces).astype(np.float32) / 32768.0
        audio = soxr.resample(mono, rate, 16000).astype(np.float32)

        window = int(cfg.vad.get("window_samples", WINDOW_SAMPLES_16K))
        threshold = float(cfg.vad.get("threshold", 0.5))
        vad = SileroVad(cfg.vad.path("model_path"), 16000)
        best = max(
            (
                vad.speech_probability(audio[i : i + window])
                for i in range(0, audio.size - window, window)
            ),
            default=0.0,
        )
        if best < threshold:
            raise RuntimeError(
                f"VAD не признал синтезированную речь речью: {best:.2f} < порога {threshold:.2f}"
            )

        segment = Segment(user_id=0, audio=audio, duration=audio.size / 16000, ended_at=0.0)
        transcript = await stt.transcribe(segment)
    finally:
        tts.close()
        stt.close()

    if transcript is None or not transcript.text.strip():
        raise RuntimeError("движок STT не расслышал синтезированную фразу")
    return f"{audio.size / 16000:.1f} с → VAD {best:.2f} → {transcript.text.strip()!r}"


# ------------------------------------------------------------- туннель ----
#
# Когда Discord закрыт по IP, работа бота целиком зависит от sing-box. Без
# этих проверок его падение всплывало только в конце отчёта строкой
# «DISCORD_TOKEN: timeout» и выглядело как проблема с токеном или сетью.

SINGBOX_CONFIG = Path("/etc/sing-box/config.json")
SOCKS_ADDRESS = os.environ.get("BASHMAK_SOCKS", "127.0.0.1:10808")
TUN_INTERFACE = os.environ.get("BASHMAK_TUN_IF", "tun-bashmak")


def _tunnel_configured() -> bool:
    return SINGBOX_CONFIG.exists()


def _run(command: list[str], timeout: float = 10.0) -> str:
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=timeout)
    except (OSError, subprocess.SubprocessError):
        return ""
    return result.stdout


def _singbox_last_error() -> str:
    """Последняя строка FATAL/ERROR из журнала: обычно в ней и причина."""
    for line in reversed(_run(
        ["journalctl", "-u", "sing-box", "-n", "30", "--no-pager", "-o", "cat"]
    ).splitlines()):
        if "FATAL" in line or "ERROR" in line:
            return line.strip()
    return ""


def _check_singbox() -> str:
    if shutil.which("systemctl") is None:
        raise RuntimeError("systemctl не найден")

    state = _run(["systemctl", "is-active", "sing-box"]).strip() or "unknown"
    if state == "active":
        return "active"

    hint = _singbox_last_error() or "подробности: journalctl -u sing-box -n 50"
    raise RuntimeError(f"{state} — {hint}")


def _check_tun() -> str:
    if not Path(f"/sys/class/net/{TUN_INTERFACE}").exists():
        raise RuntimeError(f"интерфейса {TUN_INTERFACE} нет — sing-box не поднял туннель")
    return TUN_INTERFACE


def _check_socks() -> str:
    host, _, port = SOCKS_ADDRESS.rpartition(":")
    try:
        with socket.create_connection((host, int(port)), timeout=3):
            pass
    except OSError as exc:
        raise RuntimeError(f"{SOCKS_ADDRESS} не принимает соединения ({exc})") from exc
    return f"слушает {SOCKS_ADDRESS}"


def _check_tunnel_egress() -> str:
    """Проверить, что через туннель реально ходит трафик.

    Слушающий SOCKS ничего не гарантирует: плечо до VPS может быть мёртвым, а
    в конфиге — незаменённый плейсхолдер вместо адреса. Без этой проверки
    симптом всплывает в самом низу отчёта под видом проблемы с токеном.
    """
    if shutil.which("curl") is None:
        return "curl не найден, проверка пропущена"

    result = subprocess.run(
        [
            "curl", "-sS", "--socks5-hostname", SOCKS_ADDRESS, "--max-time", "15",
            "-o", "/dev/null", "-w", "%{http_code}",
            "https://discord.com/api/v10/gateway",
        ],
        capture_output=True,
        text=True,
    )
    code = result.stdout.strip()
    if code == "200":
        return "Discord отвечает 200 через туннель"

    error = result.stderr.strip()
    hint = ""
    if "(97)" in error or "SOCKS5" in error:
        hint = " — плечо до VPS не работает: журнал sing-box покажет причину"
    raise RuntimeError(f"код {code or '—'} {error}{hint}")


def _check_dns_hijack() -> str:
    """sing-box при старте прописывает себя резолвером всего хоста.

    Домен ``~.`` на линке туннеля означает «все запросы сюда»: после этого на
    машине перестаёт работать DNS целиком, включая git и apt. tunnel.sh это
    снимает, но перехват возвращается при каждом рестарте sing-box.
    """
    if shutil.which("resolvectl") is None:
        return "resolvectl нет, проверка пропущена"

    status = _run(["resolvectl", "status", TUN_INTERFACE])
    if "~." in status:
        raise RuntimeError(
            f"{TUN_INTERFACE} перехватил DNS всего хоста. "
            f"Лечится: sudo resolvectl revert {TUN_INTERFACE}"
        )
    return "перехвата нет"


TOKEN_URL = "https://discord.com/api/v10/users/@me"


async def _check_token(token: str) -> str:
    import httpx

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(TOKEN_URL, headers={"Authorization": f"Bot {token}"})
    except httpx.TransportError as exc:
        return await _check_token_via_tunnel(token, exc)

    if response.status_code == 401:
        raise RuntimeError("Discord не принял токен (401)")
    response.raise_for_status()
    return f"бот {response.json().get('username')}"


async def _check_token_via_tunnel(token: str, direct_error: Exception) -> str:
    """Повторить проверку через локальный SOCKS туннеля.

    При блокировке Discord по IP прямой путь обязан падать: маркировка трафика
    привязана к cgroup bashmak.service, а doctor запускается из шелла, то есть
    вне туннеля. Без этого фоллбэка проверка показывала бы FAIL даже при
    полностью рабочем боте.

    Через curl, а не httpx, чтобы не тащить socksio в зависимости ради одной
    диагностической проверки. Токен передаётся в stdin, а не в argv, иначе он
    был бы виден в ps.
    """
    kind = type(direct_error).__name__
    if shutil.which("curl") is None:
        raise RuntimeError(f"напрямую {kind}, а curl для проверки через туннель не найден")

    proxy = os.environ.get("BASHMAK_SOCKS", "127.0.0.1:10808")
    options = "\n".join(
        [
            "silent",
            "show-error",
            f'socks5-hostname = "{proxy}"',
            "max-time = 15",
            'output = "/dev/null"',
            'write-out = "%{http_code}"',
            f'header = "Authorization: Bot {token}"',
            f'url = "{TOKEN_URL}"',
            "",
        ]
    )

    process = await asyncio.create_subprocess_exec(
        "curl",
        "--config",
        "-",
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout, stderr = await process.communicate(options.encode())
    code = stdout.decode(errors="replace").strip()

    if code == "200":
        return f"ок через туннель (socks {proxy}); напрямую — {kind}, это ожидаемо"
    if code == "401":
        raise RuntimeError("Discord не принял токен (401)")
    raise RuntimeError(
        f"напрямую {kind}, через туннель code={code or '—'} "
        f"{stderr.decode(errors='replace').strip()}".strip()
    )


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
    # Путь к весам лежит в подсекции активного движка: у каждого он свой, и
    # общего stt.model_path больше нет.
    stt_engine = str(cfg.stt.get("engine", "gigaam")).strip().lower()
    report.check(
        f"STT ({stt_engine})",
        lambda: _check_file(cfg.stt.get(stt_engine).path("model_path"), stt_engine),
    )
    report.check("TTS", lambda: _check_file(cfg.tts.path("model_path"), "silero-tts"))
    report.check("llama-server", lambda: _check_llama_binary(cfg.root))

    # Раньше проверок Discord: при упавшем туннеле дальше падает всё, и причина
    # видна здесь, а не в строке «DISCORD_TOKEN: timeout» внизу отчёта.
    print("\n туннель")
    if not _tunnel_configured():
        report.add(WARN, "sing-box", f"не настроен ({SINGBOX_CONFIG} нет)")
        report.add(WARN, "", "если Discord доступен напрямую — так и должно быть")
    else:
        report.check("sing-box", _check_singbox)
        report.check("интерфейс", _check_tun)
        report.check("SOCKS", _check_socks)
        report.check("выход наружу", _check_tunnel_egress)
        # Для бота не критично: он ходит через свой resolv.conf. Но ломает DNS
        # всему остальному на машине, поэтому попадает в отчёт.
        report.check("перехват DNS", _check_dns_hijack, critical=False)

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
