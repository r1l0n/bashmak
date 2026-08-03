#!/usr/bin/env bash
#
# Запуск llama-server с параметрами из config.yaml.
# Используется и вручную, и юнитом llama-server.service.
#
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PY="$ROOT/.venv/bin/python"
CONFIG="${BASHMAK_CONFIG:-$ROOT/config.yaml}"

[ -x "$PY" ]      || { echo "нет venv — сначала ./scripts/setup.sh" >&2; exit 1; }
[ -f "$CONFIG" ]  || { echo "нет $CONFIG — сначала ./scripts/setup.sh" >&2; exit 1; }

# Достаём параметры из конфига, чтобы не держать их в двух местах.
eval "$("$PY" - "$CONFIG" <<'PY'
import sys, yaml, shlex
from urllib.parse import urlparse

cfg = yaml.safe_load(open(sys.argv[1], encoding="utf-8"))
llm = cfg.get("llm", {})
url = urlparse(llm.get("server_url", "http://127.0.0.1:8080"))

for key, value in (
    ("MODEL",   llm.get("model_path", "")),
    ("THREADS", llm.get("threads", 4)),
    ("CTX",     llm.get("context_size", 4096)),
    ("HOST",    url.hostname or "127.0.0.1"),
    ("PORT",    url.port or 8080),
):
    print(f"{key}={shlex.quote(str(value))}")
PY
)"

[ -n "$MODEL" ] && [ -f "$MODEL" ] || {
    echo "модель не найдена: '$MODEL' (проверьте llm.model_path в $CONFIG)" >&2
    exit 1
}

BIN="$(find "$ROOT/vendor" -type f -name llama-server -perm -u+x 2>/dev/null | head -1)"
[ -n "$BIN" ] || { echo "llama-server не найден в vendor/ — ./scripts/setup.sh" >&2; exit 1; }

# Готовые сборки llama.cpp тянут свои .so из соседней папки.
BINDIR="$(dirname "$BIN")"
export LD_LIBRARY_PATH="$BINDIR:$BINDIR/../lib:${LD_LIBRARY_PATH:-}"

echo "llama-server: $BIN"
echo "  модель:  $MODEL"
echo "  потоки:  $THREADS, контекст: $CTX"
echo "  адрес:   http://$HOST:$PORT"

# Флаги намеренно только самые стабильные: сборка llama.cpp обновляется
# скриптом установки независимо от кода, а экзотические опции переименовываются.
exec "$BIN" \
    --model "$MODEL" \
    --host "$HOST" \
    --port "$PORT" \
    --threads "$THREADS" \
    --ctx-size "$CTX" \
    --parallel 1
