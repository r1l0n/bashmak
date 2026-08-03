#!/usr/bin/env bash
#
# Заворачивает в тоннель трафик ОДНОГО сервиса — bashmak.service.
#
# Зачем вообще: в некоторых сетях Discord заблокирован по IP (SYN дропается,
# до TLS дело не доходит), поэтому обход на уровне DPI не работает — резать
# нечего. Нужен туннель. Но заворачивать в него весь сервер незачем:
# llama-server слушает 127.0.0.1, Whisper/Piper/Silero сети не касаются
# вообще. Наружу должен ходить только сам бот.
#
# Отбор по cgroup systemd-юнита, а не по UID: не приходится заводить
# отдельного пользователя и переразбивать права на каталог проекта, venv и
# модели. Правило бьёт ровно в один сервис — ssh, apt и всё остальное
# остаётся на прямом маршруте.
#
# Вызывается из bashmak.service (ExecStartPre=+ / ExecStopPost=+), поэтому
# на момент запуска cgroup юнита уже существует — iptables сможет его
# разрезолвить.
#
#   sudo ./deploy/tunnel.sh up | down | status
#
set -euo pipefail

TUN_IF="${BASHMAK_TUN_IF:-tun-bashmak}"
TABLE="${BASHMAK_RT_TABLE:-100}"
MARK="${BASHMAK_FWMARK:-0x1}"
CGROUP="${BASHMAK_CGROUP:-system.slice/bashmak.service}"
CHAIN=BASHMAK
WAIT_IF_SECONDS=20

log() { printf '[tunnel] %s\n' "$*"; }
die() { printf '[tunnel] ОШИБКА: %s\n' "$*" >&2; exit 1; }

[ "$(id -u)" -eq 0 ] || die "нужны права root"

wait_for_interface() {
    local waited=0
    while ! ip link show "$TUN_IF" >/dev/null 2>&1; do
        if [ "$waited" -ge "$WAIT_IF_SECONDS" ]; then
            die "интерфейс $TUN_IF не появился за ${WAIT_IF_SECONDS} с.
       Проверьте: systemctl status sing-box; journalctl -u sing-box -n 50"
        fi
        sleep 1
        waited=$((waited + 1))
    done
}

up() {
    wait_for_interface

    # Ответные пакеты приходят с tun, а обратный маршрут до тех же адресов в
    # основной таблице идёт через eth0 — при строгой проверке (rp_filter=1)
    # ядро их выбросит. Нужен loose-режим. Действует max(all, интерфейс),
    # поэтому послабление именно на all неизбежно.
    sysctl -qw net.ipv4.conf.all.rp_filter=2 || true
    sysctl -qw "net.ipv4.conf.${TUN_IF}.rp_filter=2" 2>/dev/null || true

    ip route replace default dev "$TUN_IF" table "$TABLE"
    ip rule list | grep -q "fwmark ${MARK} lookup ${TABLE}" \
        || ip rule add fwmark "$MARK" lookup "$TABLE"

    iptables -t mangle -N "$CHAIN" 2>/dev/null || iptables -t mangle -F "$CHAIN"

    # Локалка — мимо тоннеля. Первым делом 127.0.0.0/8: бот ходит к
    # llama-server на 127.0.0.1:8080, и если этот трафик уедет в tun,
    # бот потеряет собственную LLM.
    iptables -t mangle -A "$CHAIN" -d 127.0.0.0/8    -j RETURN
    iptables -t mangle -A "$CHAIN" -d 10.0.0.0/8     -j RETURN
    iptables -t mangle -A "$CHAIN" -d 172.16.0.0/12  -j RETURN
    iptables -t mangle -A "$CHAIN" -d 192.168.0.0/16 -j RETURN
    # DNS НЕ исключаем: у сервиса свой resolv.conf с 1.1.1.1 (см.
    # deploy/resolv.conf и BindReadOnlyPaths в юните), и эти запросы должны
    # идти той же дорогой, что и всё остальное. Исключение сломало бы схему:
    # имена резолвились бы мимо туннеля, а то и вовсе никем.
    iptables -t mangle -A "$CHAIN" -j MARK --set-mark "$MARK"

    iptables -t mangle -C OUTPUT -m cgroup --path "$CGROUP" -j "$CHAIN" 2>/dev/null \
        || iptables -t mangle -A OUTPUT -m cgroup --path "$CGROUP" -j "$CHAIN" \
        || die "не удалось повесить правило на cgroup '$CGROUP'.
       Нужен cgroup v2 (Ubuntu 22.04+ по умолчанию) и запуск из-под юнита."

    log "трафик $CGROUP уходит в $TUN_IF (метка $MARK, таблица $TABLE)"
}

down() {
    iptables -t mangle -D OUTPUT -m cgroup --path "$CGROUP" -j "$CHAIN" 2>/dev/null || true
    iptables -t mangle -F "$CHAIN" 2>/dev/null || true
    iptables -t mangle -X "$CHAIN" 2>/dev/null || true
    ip rule del fwmark "$MARK" lookup "$TABLE" 2>/dev/null || true
    ip route flush table "$TABLE" 2>/dev/null || true
    log "маршрутизация снята"
}

status() {
    echo "интерфейс:"; ip -brief addr show "$TUN_IF" 2>/dev/null || echo "  $TUN_IF отсутствует"
    echo "правило:";   ip rule list | grep "fwmark ${MARK}" || echo "  нет"
    echo "таблица ${TABLE}:"; ip route show table "$TABLE" 2>/dev/null || echo "  пуста"
    echo "iptables:";  iptables -t mangle -S | grep -E "BASHMAK|cgroup" || echo "  нет"
}

case "${1:-}" in
    up) up ;;
    down) down ;;
    status) status ;;
    *) die "использование: $0 up|down|status" ;;
esac
