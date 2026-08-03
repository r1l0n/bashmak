#!/usr/bin/env bash
#
# Smoke-тест окружения: печатает таблицу PASS/FAIL по всем компонентам.
# Это же — приёмка «поднялось ли на новой машине».
#
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PY="$ROOT/.venv/bin/python"
if [ ! -x "$PY" ]; then
    echo "venv не найден ($PY). Сначала: ./scripts/setup.sh" >&2
    exit 1
fi

exec "$PY" -m bashmak.doctor "$@"
