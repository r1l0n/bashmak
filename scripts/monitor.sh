#!/usr/bin/env bash
#
# Монитор «Башмака»: нагрузка машины и задержки пайплайна.
# Читает logs/turns.jsonl и /proc — боту не мешает, запускать можно когда угодно.
#
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PY="$ROOT/.venv/bin/python"
[ -x "$PY" ] || { echo "нет venv — сначала ./scripts/setup.sh" >&2; exit 1; }

"$PY" -c 'import rich' 2>/dev/null || {
    echo "нет rich — ставлю" >&2
    "$PY" -m pip install -q "rich>=13.7"
}

exec "$PY" -m bashmak.monitor "$@"
