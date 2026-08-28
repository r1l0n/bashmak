"""Монитор «Башмака»: нагрузка машины и задержки пайплайна в одном окне.

Запуск: ``./scripts/monitor.sh`` или ``.venv/bin/python -m bashmak.monitor``.

Отдельный процесс: монитор нужен как раз тогда, когда с ботом что-то не так,
поэтому он не делит с ним event loop и не падает вместе с ним. Оба источника
данных пассивные:

* ``logs/turns.jsonl`` — по строке на реплику, пишет сам бот
  (bashmak/utils/logging.py);
* ``/proc`` — загрузка процессора и память.

IPC нет: бот не знает про монитор, монитор не мешает боту.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

from rich.console import Console, Group
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from .utils.logging import levels_path, metrics_path

#: Сколько последних реплик держать в статистике.
WINDOW = 60
#: Как часто перерисовывать, секунд. Чаще незачем: реплики идут секундами.
REFRESH_S = 1.0
#: Столько последних байт файла метрик читаем. Реплика — около 300 байт.
TAIL_BYTES = 256 * 1024

_SPARKS = "▁▂▃▄▅▆▇█"
#: Порядок стадий в таблице — как они идут в пайплайне, а не как пришли.
_STAGE_ORDER = ["ожидание", "stt", "intent", "очередь", "llm", "tts", "речь", "всего"]


# --------------------------------------------------------------- данные ----
def parse_turns(lines: Iterable[str]) -> list[dict[str, Any]]:
    """Разобрать JSONL, молча пропуская битые строки.

    Битая строка здесь ожидаема: монитор может прочитать файл ровно в тот
    момент, когда бот дописывает очередную реплику.
    """
    turns: list[dict[str, Any]] = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        try:
            turn = json.loads(line)
        except ValueError:
            continue
        if isinstance(turn, dict) and "stages" in turn:
            turns.append(turn)
    return turns


def read_turns(path: Path, window: int = WINDOW) -> list[dict[str, Any]]:
    """Последние реплики из файла метрик."""
    try:
        with path.open("rb") as fh:
            fh.seek(0, os.SEEK_END)
            size = fh.tell()
            fh.seek(max(0, size - TAIL_BYTES))
            chunk = fh.read()
    except OSError:
        return []
    # Первая строка после seek почти наверняка обрезана — parse_turns её отбросит.
    text = chunk.decode("utf-8", "replace")
    return parse_turns(text.splitlines())[-window:]


def percentiles(values: Sequence[float]) -> tuple[float, float, float]:
    """Медиана, 90-й процентиль и максимум.

    Среднего здесь нет: одна десятисекундная реплика смещает его настолько,
    что оно перестаёт быть показательным.
    """
    if not values:
        return 0.0, 0.0, 0.0
    ordered = sorted(values)
    p50 = ordered[len(ordered) // 2]
    p90 = ordered[min(len(ordered) - 1, int(len(ordered) * 0.9))]
    return p50, p90, ordered[-1]


def stage_stats(turns: Sequence[dict[str, Any]]) -> dict[str, tuple[float, float, float]]:
    """По каждой стадии — медиана, p90, максимум."""
    collected: dict[str, list[float]] = {}
    for turn in turns:
        for name, seconds in (turn.get("stages") or {}).items():
            collected.setdefault(name, []).append(float(seconds))
        if "total" in turn:
            collected.setdefault("всего", []).append(float(turn["total"]))

    ordered = [n for n in _STAGE_ORDER if n in collected]
    ordered += [n for n in collected if n not in _STAGE_ORDER]
    return {name: percentiles(collected[name]) for name in ordered}


def sparkline(values: Sequence[float], width: int = 60) -> str:
    """Кривая значений одной строкой."""
    if not values:
        return ""
    tail = list(values)[-width:]
    top = max(tail)
    if top <= 0:
        return _SPARKS[0] * len(tail)
    return "".join(_SPARKS[min(len(_SPARKS) - 1, int(v / top * (len(_SPARKS) - 1)))] for v in tail)


def bar(value: float, scale: float, width: int = 18) -> str:
    if scale <= 0 or value <= 0:
        return ""
    return "█" * max(1, round(width * value / scale))


# ---------------------------------------------------------------- /proc ----
def parse_meminfo(text: str) -> tuple[float, float]:
    """(занято, всего) в гигабайтах из содержимого /proc/meminfo."""
    fields: dict[str, float] = {}
    for line in text.splitlines():
        name, _, rest = line.partition(":")
        parts = rest.split()
        if parts:
            try:
                fields[name] = float(parts[0])
            except ValueError:
                continue
    total = fields.get("MemTotal", 0.0)
    available = fields.get("MemAvailable", fields.get("MemFree", 0.0))
    to_gb = 1024 * 1024
    return (total - available) / to_gb, total / to_gb


def parse_cpu_cores(text: str) -> list[tuple[float, float]]:
    """(занятое, всего) по каждому ядру из /proc/stat."""
    cores: list[tuple[float, float]] = []
    for line in text.splitlines():
        name, _, rest = line.partition(" ")
        if not (name.startswith("cpu") and name[3:].isdigit()):
            continue
        values = [float(v) for v in rest.split()]
        if len(values) < 5:
            continue
        total = sum(values)
        cores.append((total - sum(values[3:5]), total))
    return cores


def parse_cpu_times(text: str) -> tuple[float, float]:
    """(занятое, всего) из первой строки /proc/stat."""
    for line in text.splitlines():
        if not line.startswith("cpu "):
            continue
        values = [float(v) for v in line.split()[1:]]
        total = sum(values)
        idle = sum(values[3:5]) if len(values) > 4 else (values[3] if len(values) > 3 else 0.0)
        return total - idle, total
    return 0.0, 0.0


def cpu_percent(previous: tuple[float, float], current: tuple[float, float]) -> float:
    """Загрузка между двумя замерами /proc/stat."""
    busy = current[0] - previous[0]
    total = current[1] - previous[1]
    if total <= 0:
        return 0.0
    return max(0.0, min(100.0, busy / total * 100.0))


def _read(path: str) -> str:
    try:
        return Path(path).read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""


def find_process(needle: str) -> tuple[int, str] | None:
    """(pid, командная строка) процесса, в аргументах которого встречается needle."""
    proc = Path("/proc")
    if not proc.is_dir():
        return None
    for entry in proc.iterdir():
        if not entry.name.isdigit():
            continue
        try:
            raw = (entry / "cmdline").read_bytes()
        except OSError:
            continue
        cmdline = raw.decode("utf-8", "replace").replace("\0", " ").strip()
        if needle in cmdline:
            return int(entry.name), cmdline
    return None


def model_from_cmdline(cmdline: str) -> str:
    """Имя модели из строки запуска llama-server.

    Источник — процесс, а не config.yaml: конфиг отражает намерение, cmdline —
    реально загруженную модель. Значения расходятся, если сервер не
    перезапускали после правки конфига.
    """
    parts = cmdline.split()
    for index, part in enumerate(parts):
        if part == "--model" and index + 1 < len(parts):
            return Path(parts[index + 1]).stem
        if part.startswith("--model="):
            return Path(part.split("=", 1)[1]).stem
    return "?"


@dataclass
class System:
    cpu: float = 0.0
    cores: list[float] = field(default_factory=list)
    mem_used: float = 0.0
    mem_total: float = 0.0
    load: tuple[float, float, float] = (0.0, 0.0, 0.0)
    bot_pid: int | None = None
    llama_pid: int | None = None
    model: str = "?"


class SystemProbe:
    """Замеры /proc. Загрузка CPU считается между вызовами."""

    def __init__(self) -> None:
        stat = _read("/proc/stat")
        self._previous = parse_cpu_times(stat)
        self._previous_cores = parse_cpu_cores(stat)

    def sample(self) -> System:
        stat = _read("/proc/stat")
        current = parse_cpu_times(stat)
        cpu = cpu_percent(self._previous, current)
        self._previous = current

        cores_now = parse_cpu_cores(stat)
        cores = [
            cpu_percent(before, after)
            for before, after in zip(self._previous_cores, cores_now)
        ]
        self._previous_cores = cores_now

        used, total = parse_meminfo(_read("/proc/meminfo"))
        try:
            load = tuple(float(v) for v in _read("/proc/loadavg").split()[:3])  # type: ignore[assignment]
        except (ValueError, IndexError):
            load = (0.0, 0.0, 0.0)

        # Юнит запускает бота как `python -m bashmak.bot` — по этому и ищем.
        bot = find_process("bashmak.bot")
        llama = find_process("llama-server")
        return System(
            cpu=cpu,
            cores=cores,
            mem_used=used,
            mem_total=total,
            load=load if len(load) == 3 else (0.0, 0.0, 0.0),
            bot_pid=bot[0] if bot else None,
            llama_pid=llama[0] if llama else None,
            model=model_from_cmdline(llama[1]) if llama else "?",
        )


# ------------------------------------------------------------- отрисовка ----
def _health(label: str, pid: int | None) -> Text:
    if pid is None:
        return Text.assemble((f"{label} ", "dim"), ("не найден", "bold red"))
    return Text.assemble((f"{label} ", "dim"), (f"жив ({pid})", "bold green"))


def _meter(value: float, scale: float, width: int = 16) -> Text:
    filled = 0 if scale <= 0 else max(0, min(width, round(width * value / scale)))
    colour = "green" if value < scale * 0.6 else "yellow" if value < scale * 0.85 else "red"
    return Text.assemble(("█" * filled, colour), ("░" * (width - filled), "dim"))


def render_system(system: System, turns: Sequence[dict[str, Any]]) -> Panel:
    recent_hour = sum(1 for t in turns if time.time() - t.get("at", 0) < 3600)

    line1 = Text.assemble(
        ("CPU ", "dim"),
        _meter(system.cpu, 100.0),
        (f" {system.cpu:5.1f}%   ", "bold"),
        ("RAM ", "dim"),
        _meter(system.mem_used, system.mem_total or 1.0),
        (f" {system.mem_used:.1f}/{system.mem_total:.1f} ГБ   ", "bold"),
        ("LA ", "dim"),
        (" ".join(f"{v:.2f}" for v in system.load), "bold"),
    )
    line2 = Text.assemble(
        _health("llama-server", system.llama_pid),
        ("   ", ""),
        _health("бот", system.bot_pid),
        ("   реплик за час: ", "dim"),
        (str(recent_hour), "bold"),
    )
    line3 = Text.assemble(("модель ", "dim"), (system.model, "bold cyan"))
    return Panel(Group(line1, line2, line3), title="Башмак", border_style="cyan")


def render_cores(cores: Sequence[float], per_row: int = 2, width: int = 8) -> Panel:
    """Загрузка по ядрам — по колонке на каждое."""
    table = Table(box=None, pad_edge=False, expand=True, show_header=False)
    for _ in range(per_row):
        table.add_column("", no_wrap=True)

    cells: list[Text] = []
    for index, value in enumerate(cores):
        filled = max(0, min(width, round(value / 100 * width)))
        colour = "green" if value < 60 else "yellow" if value < 85 else "red"
        cells.append(
            Text.assemble(
                (f"{index:>2} ", "dim"),
                ("█" * filled, colour),
                ("░" * (width - filled), "dim"),
                (f"{value:>4.0f}%", ""),
            )
        )
    if not cells:
        cells = [Text("нет данных (/proc недоступен)", style="dim")]

    for start in range(0, len(cells), per_row):
        row = cells[start : start + per_row]
        table.add_row(*(row + [""] * (per_row - len(row))))
    return Panel(table, title=f"ядра ({len(cores)})", border_style="magenta")


def render_stages(turns: Sequence[dict[str, Any]]) -> Panel:
    stats = stage_stats(turns)
    table = Table(box=None, pad_edge=False, expand=True)
    table.add_column("стадия", style="bold")
    table.add_column("медиана", justify="right")
    table.add_column("p90", justify="right")
    table.add_column("макс", justify="right")
    table.add_column("", ratio=1)

    scale = max((v[1] for v in stats.values()), default=1.0)
    for name, (p50, p90, top) in stats.items():
        style = "cyan" if name == "всего" else "white"
        table.add_row(
            Text(name, style=style),
            f"{p50:.1f}",
            f"{p90:.1f}",
            f"{top:.1f}",
            # 12 клеток, а не шире: длиннее — и полоса не влезает в колонку,
            # rich обрезает её многоточием, и сравнивать становится нечего.
            Text(bar(p90, scale, width=12), style="magenta" if name == "llm" else "blue"),
        )
    if not stats:
        table.add_row("нет данных", "", "", "", "")
    return Panel(table, title=f"задержки, с ({len(turns)} реплик)", border_style="blue")


def render_turns(turns: Sequence[dict[str, Any]], limit: int = 6) -> Panel:
    table = Table(box=None, pad_edge=False, expand=True, show_header=False)
    table.add_column("", width=5, style="dim")
    # С переносом: ответ нужен целиком, ради этого показываем меньше реплик.
    # По обрезанному ответу не понять, что бот сказал.
    table.add_column("", ratio=1)
    table.add_column("", width=6, justify="right")

    for turn in list(turns)[-limit:]:
        stamp = time.strftime("%H:%M", time.localtime(turn.get("at", time.time())))
        speaker = turn.get("speaker") or "?"
        heard = turn.get("heard") or ""
        reply = turn.get("reply") or ""
        body = Text.assemble((f"{speaker}: ", "bold"), (heard, ""), ("\n  → ", "dim"), (reply, "green"))
        table.add_row(stamp, body, f"{turn.get('total', 0.0):.1f}с")
    if not turns:
        table.add_row("", Text("реплик ещё не было", style="dim"), "")
    return Panel(table, title="последние реплики", border_style="green")


def read_levels(path: Path) -> dict[str, Any]:
    """Снимок уровней. Пустой словарь, если бот их не пишет."""
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return payload if isinstance(payload, dict) else {}


def render_levels(levels: dict[str, Any], width: int = 22) -> Panel:
    """Полосы уровня по говорящим — кто говорит и слышит ли его VAD."""
    users = levels.get("users") or []
    threshold = float(levels.get("threshold", 0.5))
    stale = time.time() - float(levels.get("at", 0)) > 3.0

    table = Table(box=None, pad_edge=False, expand=True, show_header=False)
    table.add_column("", ratio=1, no_wrap=True, overflow="ellipsis")
    table.add_column("", width=width)
    table.add_column("", width=4, justify="right")

    if not users:
        table.add_row(Text("в канале тихо", style="dim"), "", "")
    for user in sorted(users, key=lambda u: -float(u.get("peak", 0))):
        peak = float(user.get("peak", 0.0))
        vad = float(user.get("vad", 0.0))
        filled = max(0, min(width, round(peak * width)))
        # Цвет по оценке VAD, а не по громкости: важно не «шумно», а
        # «распознаётся ли это как речь».
        colour = "green" if user.get("speech") else "yellow" if vad >= threshold else "blue"
        table.add_row(
            Text(str(user.get("name", "?")), style="bold" if user.get("speech") else ""),
            Text.assemble(("█" * filled, colour), ("░" * (width - filled), "dim")),
            Text(f"{vad:.2f}", style="dim" if vad < threshold else colour),
        )

    title = "уровни" if not stale else "уровни (бот молчит)"
    return Panel(table, title=f"{title}  порог VAD {threshold:.2f}", border_style="yellow")


def render_trend(turns: Sequence[dict[str, Any]]) -> Panel:
    totals = [float(t.get("total", 0.0)) for t in turns]
    line = sparkline(totals) or "нет данных"
    top = max(totals, default=0.0)
    return Panel(
        Text.assemble((line, "cyan"), (f"   макс {top:.1f} с", "dim")),
        title="время ответа",
        border_style="cyan",
    )


def build(
    system: System, turns: Sequence[dict[str, Any]], levels: dict[str, Any] | None = None
) -> Layout:
    cores = system.cores
    layout = Layout()
    layout.split_column(
        Layout(render_system(system, turns), size=5),
        Layout(name="middle", ratio=1),
        Layout(render_trend(turns), size=3),
    )
    layout["middle"].split_row(
        Layout(name="left", ratio=2),
        Layout(render_turns(turns), ratio=3),
    )
    layout["left"].split_column(
        Layout(render_stages(turns), size=10),
        # По две колонки в ряд плюс рамка — столько строк панель и займёт.
        Layout(render_cores(cores), size=(len(cores) + 1) // 2 + 2),
        Layout(render_levels(levels or {})),
    )
    return layout


def main() -> None:
    try:
        from .config import load_config

        cfg = load_config()
        path, levels = metrics_path(cfg), levels_path(cfg)
    except Exception:
        # Конфига может не быть — монитор всё равно должен запуститься.
        path, levels = metrics_path(), levels_path()

    console = Console()
    probe = SystemProbe()
    console.print(f"[dim]метрики: {path}[/dim]")

    try:
        with Live(console=console, screen=True, refresh_per_second=4) as live:
            while True:
                live.update(build(probe.sample(), read_turns(path), read_levels(levels)))
                time.sleep(REFRESH_S)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
