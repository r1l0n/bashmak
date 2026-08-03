#!/usr/bin/env bash
#
# Полная подготовка окружения «Башмака» на чистой Ubuntu.
# После `git clone` достаточно запустить этот скрипт — он поставит системные
# пакеты, соберёт venv, притащит llama.cpp и скачает все веса моделей.
#
# Скрипт идемпотентный: повторный запуск ничего не ломает и не перекачивает.
#
#   ./scripts/setup.sh                      # всё, профиль balanced
#   ./scripts/setup.sh --profile fast       # модели полегче
#   ./scripts/setup.sh --skip-models        # только окружение
#   ./scripts/setup.sh --models-only        # только докачать модели
#   ./scripts/setup.sh --systemd            # + установить и включить юниты
#   ./scripts/setup.sh --tunnel             # + пустить трафик бота через sing-box
#   ./scripts/setup.sh --force              # перекачать/пересобрать всё
#
# Юниты генерируются из deploy/*.template при установке. Если шаблон поменялся
# в репозитории, перезапустите с --systemd, иначе в /etc/systemd/system
# останется копия времён прошлой установки.
#
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

# ---------------------------------------------------------------- вывод ----
if [ -t 1 ]; then
    C_OK=$'\033[32m'; C_SKIP=$'\033[90m'; C_WARN=$'\033[33m'
    C_ERR=$'\033[31m'; C_HEAD=$'\033[1;36m'; C_OFF=$'\033[0m'
else
    C_OK=''; C_SKIP=''; C_WARN=''; C_ERR=''; C_HEAD=''; C_OFF=''
fi

step() { printf '\n%s==> %s%s\n' "$C_HEAD" "$*" "$C_OFF"; }
ok()   { printf '    %s[ok]%s   %s\n'   "$C_OK"   "$C_OFF" "$*"; }
skip() { printf '    %s[skip]%s %s\n'   "$C_SKIP" "$C_OFF" "$*"; }
warn() { printf '    %s[warn]%s %s\n'   "$C_WARN" "$C_OFF" "$*" >&2; }
die()  { printf '\n%s[fail]%s %s\n\n'   "$C_ERR"  "$C_OFF" "$*" >&2; exit 1; }

# ---------------------------------------------------------------- флаги ----
PROFILE=balanced
DO_MODELS=1
DO_ENV=1
DO_SYSTEMD=0
DO_TUNNEL=0
FORCE=0

while [ $# -gt 0 ]; do
    case "$1" in
        --profile)     PROFILE="${2:-}"; shift 2 ;;
        --profile=*)   PROFILE="${1#*=}"; shift ;;
        --skip-models) DO_MODELS=0; shift ;;
        --models-only) DO_ENV=0; shift ;;
        --systemd)     DO_SYSTEMD=1; shift ;;
        --tunnel)      DO_TUNNEL=1; DO_SYSTEMD=1; shift ;;
        --force)       FORCE=1; shift ;;
        -h|--help)     sed -n '2,25p' "$0"; exit 0 ;;
        *)             die "неизвестный флаг: $1 (см. --help)" ;;
    esac
done

case "$PROFILE" in
    fast)
        LLM_REPO="bartowski/Qwen2.5-3B-Instruct-GGUF"
        LLM_FILE="Qwen2.5-3B-Instruct-Q4_K_M.gguf"
        LLM_REPO_ALT="Qwen/Qwen2.5-3B-Instruct-GGUF"
        STT_REPO="Systran/faster-whisper-small"
        ;;
    balanced)
        LLM_REPO="bartowski/Qwen2.5-7B-Instruct-GGUF"
        LLM_FILE="Qwen2.5-7B-Instruct-Q4_K_M.gguf"
        LLM_REPO_ALT="Qwen/Qwen2.5-7B-Instruct-GGUF"
        STT_REPO="Systran/faster-whisper-small"
        ;;
    quality)
        LLM_REPO="bartowski/Qwen2.5-7B-Instruct-GGUF"
        LLM_FILE="Qwen2.5-7B-Instruct-Q4_K_M.gguf"
        LLM_REPO_ALT="Qwen/Qwen2.5-7B-Instruct-GGUF"
        STT_REPO="Systran/faster-whisper-medium"
        ;;
    *) die "неизвестный профиль '$PROFILE' (fast | balanced | quality)" ;;
esac

STT_DIR="models/stt/$(basename "$STT_REPO")"
TTS_VOICE="ru_RU-irina-medium"
VAD_URL="https://raw.githubusercontent.com/snakers4/silero-vad/master/src/silero_vad/data/silero_vad.onnx"
VENV="$ROOT/.venv"
PY="$VENV/bin/python"

printf '%s\n' "$C_HEAD"
printf 'Башмак — подготовка окружения\n'
printf '  корень:  %s\n' "$ROOT"
printf '  профиль: %s\n' "$PROFILE"
printf '%s' "$C_OFF"

# ------------------------------------------------------ 1. проверки ОС ----
check_host() {
    step "1/8  Проверка системы"

    [ "$(uname -s)" = "Linux" ] || die "скрипт рассчитан на Linux (цель — ubuntu-srv)"
    [ "$(uname -m)" = "x86_64" ] || die "нужен x86_64 (готовые сборки llama.cpp только под него)"

    if [ -r /etc/os-release ]; then
        # shellcheck disable=SC1091
        . /etc/os-release
        case "${ID:-}${ID_LIKE:-}" in
            *debian*|*ubuntu*) ok "ОС: ${PRETTY_NAME:-unknown}" ;;
            *) warn "ОС ${PRETTY_NAME:-unknown} не Debian-подобная — шаг с apt придётся сделать руками" ;;
        esac
    fi

    if [ "$(id -u)" -eq 0 ]; then
        SUDO=""
    elif command -v sudo >/dev/null 2>&1; then
        SUDO="sudo"
    else
        die "нужен root или sudo для установки системных пакетов"
    fi

    # Если репозиторий пушили с Windows, бит исполняемости в индексе не
    # сохранился и systemd не смог бы запустить run_llama_server.sh.
    if chmod +x "$ROOT"/scripts/*.sh "$ROOT"/deploy/*.sh 2>/dev/null; then
        ok "скрипты помечены исполняемыми"
    fi

    local avail
    avail="$(df -P -BG "$ROOT" | awk 'NR==2 {gsub(/G/,"",$4); print $4}')"
    if [ -n "$avail" ] && [ "$avail" -lt 20 ]; then
        die "мало места: ${avail} ГБ свободно, нужно минимум 20 ГБ (модели ~7 ГБ + сборки)"
    fi
    ok "свободно ${avail:-?} ГБ"
}

# ------------------------------------------------- 2. системные пакеты ----
APT_PACKAGES=(
    python3-venv python3-dev build-essential cmake pkg-config
    git curl jq unzip tar ca-certificates
    ffmpeg libopus0 libopus-dev libsodium-dev
)

install_apt() {
    step "2/8  Системные пакеты"

    local missing=()
    for p in "${APT_PACKAGES[@]}"; do
        dpkg -s "$p" >/dev/null 2>&1 || missing+=("$p")
    done

    if [ ${#missing[@]} -eq 0 ]; then
        skip "все пакеты уже стоят"
    else
        ok "ставлю: ${missing[*]}"
        $SUDO apt-get update -qq
        DEBIAN_FRONTEND=noninteractive $SUDO apt-get install -y -qq "${missing[@]}"
        ok "установлено"
    fi

    # ffmpeg и libopus — не «желательно», а обязательно:
    # без ffmpeg не будет музыки, без opus py-cord не сможет ни принять, ни отдать голос.
    command -v ffmpeg >/dev/null || die "ffmpeg не найден после установки"
    ok "ffmpeg: $(ffmpeg -version 2>/dev/null | head -1 | cut -d' ' -f1-3)"
}

# --------------------------------------------------------- 3. python ----
pick_python() {
    local c v
    for c in python3.12 python3.11 python3.10; do
        command -v "$c" >/dev/null 2>&1 && { echo "$c"; return 0; }
    done
    if command -v python3 >/dev/null 2>&1; then
        v="$(python3 -c 'import sys; print(sys.version_info[0]*100+sys.version_info[1])')"
        if [ "$v" -ge 310 ] && [ "$v" -le 312 ]; then echo python3; return 0; fi
        echo "__too_new__:$(python3 -c 'import sys; print("%d.%d"%sys.version_info[:2])')"
        return 0
    fi
    return 1
}

setup_venv() {
    step "3/8  Python-окружение"

    local base
    base="$(pick_python)" || die "python3 не найден"

    case "$base" in
        __too_new__:*)
            die "системный python — ${base#*:}, а колёс ctranslate2/onnxruntime под 3.13+ ещё нет.
       Поставьте 3.12 и перезапустите:
           sudo add-apt-repository -y ppa:deadsnakes/ppa
           sudo apt-get update && sudo apt-get install -y python3.12 python3.12-venv python3.12-dev"
            ;;
    esac
    ok "интерпретатор: $base ($("$base" -V 2>&1))"

    if [ "$FORCE" -eq 1 ] && [ -d "$VENV" ]; then
        rm -rf "$VENV"
        ok "старый venv удалён (--force)"
    fi

    if [ -x "$PY" ]; then
        skip "venv уже есть: $VENV"
    else
        "$base" -m venv "$VENV"
        ok "venv создан: $VENV"
    fi

    "$PY" -m pip install --quiet --upgrade pip wheel setuptools
    ok "pip обновлён"

    # requirements не пере-резолвится каждый раз: сверяем хеш файла.
    local stamp="$VENV/.requirements.sha256" now
    now="$(sha256sum requirements.txt | cut -d' ' -f1)"
    if [ "$FORCE" -eq 0 ] && [ -f "$stamp" ] && [ "$(cat "$stamp")" = "$now" ]; then
        skip "зависимости не менялись"
    else
        ok "ставлю зависимости (это надолго — ctranslate2/onnxruntime тяжёлые)"
        "$PY" -m pip install --upgrade -r requirements.txt
        echo "$now" > "$stamp"
        ok "зависимости установлены"
    fi
}

# ------------------------------------------------------ 4. llama.cpp ----
LLAMA_DIR="$ROOT/vendor/llama.cpp"

find_llama_server() {
    find "$LLAMA_DIR" -type f -name llama-server -perm -u+x 2>/dev/null | head -1
}

build_llama_from_source() {
    warn "готовой сборки нет — собираю из исходников (5–15 минут)"
    local src="$ROOT/vendor/llama.cpp-src"
    if [ -d "$src/.git" ]; then
        git -C "$src" fetch --depth 1 origin master && git -C "$src" reset --hard FETCH_HEAD
    else
        rm -rf "$src"
        git clone --depth 1 https://github.com/ggml-org/llama.cpp "$src"
    fi
    cmake -S "$src" -B "$src/build" -DCMAKE_BUILD_TYPE=Release -DGGML_NATIVE=ON -DLLAMA_CURL=OFF
    cmake --build "$src/build" --target llama-server llama-gguf-split -j "$(nproc)"
    mkdir -p "$LLAMA_DIR"
    cp -r "$src/build/bin/." "$LLAMA_DIR/"
}

setup_llama() {
    step "4/8  llama.cpp"

    if [ "$FORCE" -eq 1 ]; then
        rm -rf "$LLAMA_DIR"
    elif [ -n "$(find_llama_server)" ]; then
        skip "llama-server уже собран: $(find_llama_server)"
        return
    fi

    mkdir -p "$LLAMA_DIR"

    local url tmp
    url="$(curl -fsSL https://api.github.com/repos/ggml-org/llama.cpp/releases/latest \
          | jq -r '.assets[] | select(.name | test("^llama-.*-bin-ubuntu-x64\\.(zip|tar\\.gz)$")) | .browser_download_url' \
          | head -1)" || url=""

    if [ -z "$url" ] || [ "$url" = "null" ]; then
        build_llama_from_source
    else
        ok "качаю $(basename "$url")"
        tmp="$(mktemp -d)"
        trap 'rm -rf "$tmp"' RETURN
        curl -fsSL -o "$tmp/llama.archive" "$url"
        case "$url" in
            *.zip)     unzip -q "$tmp/llama.archive" -d "$LLAMA_DIR" ;;
            *.tar.gz)  tar -xzf "$tmp/llama.archive" -C "$LLAMA_DIR" ;;
        esac
        # В zip-ассетах бит исполняемости не сохраняется.
        find "$LLAMA_DIR" -type f -name 'llama-*' -exec chmod +x {} + 2>/dev/null || true
    fi

    local bin
    bin="$(find_llama_server)"
    [ -n "$bin" ] || die "llama-server не найден после установки — смотрите $LLAMA_DIR"

    # Готовые сборки линкуются на свои .so рядом — без LD_LIBRARY_PATH бинарник не стартует.
    local libdir
    libdir="$(dirname "$bin")"
    if ! LD_LIBRARY_PATH="$libdir:$libdir/../lib:${LD_LIBRARY_PATH:-}" "$bin" --version >/dev/null 2>&1; then
        warn "llama-server --version отработал с ошибкой — проверьте вручную: $bin"
    fi
    ok "llama-server: $bin"
}

# --------------------------------------------------------- 5. модели ----
hf_get_file() {
    # repo, filename, dest_dir -> путь к файлу
    "$PY" - "$1" "$2" "$3" <<'PY'
import sys
from huggingface_hub import hf_hub_download
repo, fname, dest = sys.argv[1:4]
print(hf_hub_download(repo_id=repo, filename=fname, local_dir=dest))
PY
}

hf_get_repo() {
    # repo, dest_dir — только то, что нужно ctranslate2
    "$PY" - "$1" "$2" <<'PY'
import sys
from huggingface_hub import snapshot_download
repo, dest = sys.argv[1:3]
print(snapshot_download(
    repo_id=repo,
    local_dir=dest,
    allow_patterns=["*.bin", "*.json", "*.txt", "*.model"],
))
PY
}

fetch_llm() {
    local target="models/llm/$LLM_FILE"
    if [ "$FORCE" -eq 0 ] && [ -s "$target" ]; then
        skip "LLM уже на месте ($(du -h "$target" | cut -f1))"
        return
    fi
    mkdir -p models/llm
    ok "качаю LLM $LLM_REPO / $LLM_FILE"
    if ! hf_get_file "$LLM_REPO" "$LLM_FILE" "models/llm" >/dev/null; then
        # Официальный репозиторий Qwen раздаёт Q4_K_M шардами (-00001-of-0000N.gguf).
        # Современный llama.cpp грузит их по первому шарду, но склеим явно —
        # чтобы конфиг указывал на один предсказуемый файл.
        warn "не вышло с $LLM_REPO, пробую $LLM_REPO_ALT (шардированный)"
        local first
        first="$("$PY" - "$LLM_REPO_ALT" "models/llm" <<'PY'
import sys, glob, os
from huggingface_hub import snapshot_download
repo, dest = sys.argv[1:3]
snapshot_download(repo_id=repo, local_dir=dest, allow_patterns=["*q4_k_m*.gguf", "*Q4_K_M*.gguf"])
shards = sorted(glob.glob(os.path.join(dest, "*[qQ]4_[kK]_[mM]*.gguf")))
print(shards[0] if shards else "")
PY
)"
        [ -n "$first" ] || die "не удалось скачать GGUF ни из $LLM_REPO, ни из $LLM_REPO_ALT"
        case "$first" in
            *-00001-of-*)
                local split_bin
                split_bin="$(find "$LLAMA_DIR" -type f -name llama-gguf-split | head -1)"
                if [ -n "$split_bin" ]; then
                    LD_LIBRARY_PATH="$(dirname "$split_bin"):$(dirname "$split_bin")/../lib:${LD_LIBRARY_PATH:-}" \
                        "$split_bin" --merge "$first" "$target"
                    ok "шарды склеены в $target"
                else
                    warn "llama-gguf-split не найден — оставляю шарды, укажите в config.yaml: $first"
                    target="$first"
                fi
                ;;
            *) mv -f "$first" "$target" ;;
        esac
    fi
    ok "LLM: $target ($(du -h "$target" 2>/dev/null | cut -f1))"
}

fetch_stt() {
    if [ "$FORCE" -eq 0 ] && [ -s "$STT_DIR/model.bin" ]; then
        skip "STT уже на месте ($STT_DIR)"
        return
    fi
    ok "качаю STT $STT_REPO"
    hf_get_repo "$STT_REPO" "$STT_DIR" >/dev/null
    [ -s "$STT_DIR/model.bin" ] || die "в $STT_DIR нет model.bin"
    ok "STT: $STT_DIR"
}

fetch_tts() {
    if [ "$FORCE" -eq 0 ] && [ -s "models/tts/$TTS_VOICE.onnx" ]; then
        skip "голос Piper уже на месте"
        return
    fi
    mkdir -p models/tts
    ok "качаю голос Piper $TTS_VOICE"
    "$PY" -m piper.download_voices "$TTS_VOICE" --data-dir models/tts
    [ -s "models/tts/$TTS_VOICE.onnx" ] || die "голос $TTS_VOICE не скачался"
    ok "TTS: models/tts/$TTS_VOICE.onnx"
}

fetch_vad() {
    if [ "$FORCE" -eq 0 ] && [ -s models/vad/silero_vad.onnx ]; then
        skip "Silero VAD уже на месте"
        return
    fi
    mkdir -p models/vad
    ok "качаю Silero VAD (2 МБ)"
    curl -fsSL -o models/vad/silero_vad.onnx "$VAD_URL"
    ok "VAD: models/vad/silero_vad.onnx"
}

fetch_models() {
    step "5/8  Веса моделей (профиль: $PROFILE)"
    [ -x "$PY" ] || die "venv не готов — запустите без --models-only"
    fetch_vad
    fetch_llm
    fetch_stt
    fetch_tts
}

# --------------------------------------------------------- 6. конфиг ----
setup_config() {
    step "6/8  Конфигурация"

    if [ -f config.yaml ]; then
        skip "config.yaml уже есть — не трогаю"
    else
        cp config.example.yaml config.yaml
        # Подставляем пути и профиль под то, что реально скачали.
        sed -i \
            -e "s|^profile: .*|profile: $PROFILE|" \
            -e "s|^  model_path: models/llm/.*|  model_path: models/llm/$LLM_FILE|" \
            -e "s|^  model_path: models/stt/.*|  model_path: $STT_DIR|" \
            -e "s|^  voice_path: models/tts/.*|  voice_path: models/tts/$TTS_VOICE.onnx|" \
            config.yaml
        ok "config.yaml создан из шаблона"
    fi

    mkdir -p logs
    if [ -f .env ]; then
        if grep -qE '^DISCORD_TOKEN=.+' .env; then
            ok ".env на месте, токен задан"
        else
            warn ".env есть, но DISCORD_TOKEN пуст — бот не запустится"
        fi
    else
        cp .env.example .env
        warn ".env создан из шаблона. ВПИШИТЕ DISCORD_TOKEN перед запуском!"
    fi
}

# -------------------------------------------------------- 7. systemd ----
setup_systemd() {
    step "7/8  systemd"

    if [ "$DO_SYSTEMD" -eq 0 ]; then
        skip "пропущено (добавьте --systemd, чтобы поставить юниты)"
        return
    fi

    if [ "$DO_TUNNEL" -eq 1 ] && [ ! -x /usr/local/bin/sing-box ]; then
        warn "запрошен --tunnel, но sing-box не установлен."
        warn "  сначала: sudo ./scripts/install_singbox.sh"
    fi

    local user unit
    user="$(id -un)"
    for unit in llama-server bashmak; do
        sed -e "s|@@ROOT@@|$ROOT|g" -e "s|@@USER@@|$user|g" \
            "deploy/$unit.service.template" > "/tmp/$unit.service"

        # В шаблоне строки туннеля закомментированы: он нужен только там,
        # где Discord режется по IP. --tunnel их раскомментирует.
        if [ "$unit" = bashmak ] && [ "$DO_TUNNEL" -eq 1 ]; then
            sed -i \
                -e 's|^#Requires=sing-box|Requires=sing-box|' \
                -e 's|^#After=sing-box|After=sing-box|' \
                -e 's|^#ExecStartPre=+|ExecStartPre=+|' \
                -e 's|^#ExecStopPost=+|ExecStopPost=+|' \
                "/tmp/$unit.service"
            ok "bashmak.service: трафик пойдёт через sing-box"
        fi

        $SUDO install -m 0644 "/tmp/$unit.service" "/etc/systemd/system/$unit.service"
        rm -f "/tmp/$unit.service"
        ok "/etc/systemd/system/$unit.service"
    done

    $SUDO systemctl daemon-reload
    $SUDO systemctl enable llama-server.service bashmak.service >/dev/null
    ok "юниты включены (автозапуск при загрузке)"

    # enable — это только автозапуск при загрузке. Без явного start сервис
    # так и останется в состоянии «inactive (dead)», а doctor.sh покажет,
    # что llama-server не отвечает.
    $SUDO systemctl restart llama-server.service
    ok "llama-server запущен, жду загрузки модели в память"

    local waited=0
    while [ "$waited" -lt 300 ]; do
        if curl -fsS --max-time 2 "http://127.0.0.1:8080/health" >/dev/null 2>&1; then
            ok "llama-server отвечает (${waited} с)"
            break
        fi
        if ! $SUDO systemctl is-active --quiet llama-server.service; then
            warn "llama-server упал — смотрите: journalctl -u llama-server -n 50"
            break
        fi
        sleep 3
        waited=$((waited + 3))
    done
    [ "$waited" -lt 300 ] || warn "llama-server не поднялся за 5 минут"

    if grep -qE '^DISCORD_TOKEN=.+' "$ROOT/.env" 2>/dev/null; then
        $SUDO systemctl restart bashmak.service
        ok "бот запущен: journalctl -u bashmak -f"
    else
        warn "бот не запущен: сначала впишите DISCORD_TOKEN в .env, потом"
        warn "  sudo systemctl start bashmak"
    fi
}

# ---------------------------------------------------------- 8. итог ----
summary() {
    step "8/8  Готово"
    cat <<EOF

    Проверить окружение:   ./scripts/doctor.sh
    Запустить вручную:     ./scripts/run_llama_server.sh &   и   .venv/bin/python -m bashmak.bot
    Запустить как сервис:  sudo systemctl start bashmak

EOF
    if ! grep -qE '^DISCORD_TOKEN=.+' .env 2>/dev/null; then
        warn "СНАЧАЛА впишите DISCORD_TOKEN в $ROOT/.env"
    fi
}

# ------------------------------------------------------------- main ----
check_host
if [ "$DO_ENV" -eq 1 ]; then
    install_apt
    setup_venv
    setup_llama
else
    skip "окружение пропущено (--models-only)"
fi

if [ "$DO_MODELS" -eq 1 ]; then
    fetch_models
else
    step "5/8  Веса моделей"
    skip "пропущено (--skip-models)"
fi

setup_config
setup_systemd
summary
