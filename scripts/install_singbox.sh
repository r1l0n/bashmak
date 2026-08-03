#!/usr/bin/env bash
#
# Установка sing-box из релиза на GitHub.
#
# Через apt (deb.sagernet.org) поставить получается не везде: репозиторий
# живёт на AWS и в заблокированных сетях отваливается по таймауту ровно так
# же, как сам Discord. GitHub при этом обычно доступен — оттуда и берём.
#
# Ставит бинарник в /usr/local/bin, юнит в systemd и заготовку конфига в
# /etc/sing-box. Сервис НЕ запускается: в конфиге плейсхолдеры, их сперва
# надо заменить своими значениями.
#
#   sudo ./scripts/install_singbox.sh [версия]
#
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VERSION="${1:-}"
BIN=/usr/local/bin/sing-box

if [ -t 1 ]; then
    C_OK=$'\033[32m'; C_WARN=$'\033[33m'; C_ERR=$'\033[31m'; C_OFF=$'\033[0m'
else
    C_OK=''; C_WARN=''; C_ERR=''; C_OFF=''
fi
ok()   { printf '    %s[ok]%s   %s\n' "$C_OK" "$C_OFF" "$*"; }
warn() { printf '    %s[warn]%s %s\n' "$C_WARN" "$C_OFF" "$*" >&2; }
die()  { printf '\n%s[fail]%s %s\n\n' "$C_ERR" "$C_OFF" "$*" >&2; exit 1; }

[ "$(id -u)" -eq 0 ] || die "запускайте через sudo"
[ "$(uname -m)" = "x86_64" ] || die "скрипт рассчитан на x86_64"

printf '\n==> Установка sing-box\n'

# Сломанный apt-источник только мешает: каждый apt update будет висеть на нём
# по минуте. Раз ставим бинарником — источник не нужен.
if [ -f /etc/apt/sources.list.d/sagernet.list ]; then
    rm -f /etc/apt/sources.list.d/sagernet.list
    ok "убран нерабочий apt-источник deb.sagernet.org"
fi

if [ -z "$VERSION" ]; then
    url="$(curl -fsSL --max-time 30 https://api.github.com/repos/SagerNet/sing-box/releases/latest \
        | jq -r '.assets[] | select(.name | test("linux-amd64\\.tar\\.gz$")) | .browser_download_url' \
        | head -1)" || url=""
else
    url="https://github.com/SagerNet/sing-box/releases/download/v${VERSION}/sing-box-${VERSION}-linux-amd64.tar.gz"
fi

[ -n "$url" ] && [ "$url" != "null" ] \
    || die "не удалось узнать адрес релиза. Укажите версию явно:
       sudo $0 1.11.15
       Список: https://github.com/SagerNet/sing-box/releases"

ok "качаю $(basename "$url")"
tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT
curl -fsSL --max-time 300 -o "$tmp/sing-box.tar.gz" "$url" \
    || die "не скачалось: $url"
tar -xzf "$tmp/sing-box.tar.gz" -C "$tmp"

src="$(find "$tmp" -type f -name sing-box -perm -u+x | head -1)"
[ -n "$src" ] || die "в архиве нет бинарника sing-box"
install -m 0755 "$src" "$BIN"
ok "$BIN — $("$BIN" version | head -1)"

mkdir -p /etc/sing-box /var/lib/sing-box
if [ -f /etc/sing-box/config.json ]; then
    ok "/etc/sing-box/config.json уже есть — не трогаю"
else
    install -m 0600 "$ROOT/deploy/sing-box.json.example" /etc/sing-box/config.json
    warn "создан /etc/sing-box/config.json из шаблона — ЗАМЕНИТЕ плейсхолдеры"
fi

install -m 0644 "$ROOT/deploy/sing-box.service" /etc/systemd/system/sing-box.service
systemctl daemon-reload
ok "юнит sing-box.service установлен"

cat <<EOF

    Дальше:
      1. sudo nano /etc/sing-box/config.json      — вписать адрес, uuid, public_key
      2. sudo sing-box check -c /etc/sing-box/config.json
      3. sudo systemctl enable --now sing-box
      4. curl -sS --socks5-hostname 127.0.0.1:10808 --max-time 15 \\
             -o /dev/null -w 'code=%{http_code}\\n' https://discord.com/api/v10/gateway

EOF
