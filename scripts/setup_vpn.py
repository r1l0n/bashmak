#!/usr/bin/env python3
"""Настройка туннеля sing-box: `sudo ./scripts/setup_vpn.py`.

Собирает /etc/sing-box/config.json из шаблона deploy/sing-box.json.example,
подставляя параметры подключения. Всё, что специфично для конкретной VPS
(адрес, uuid, ключ Reality), спрашивается здесь и в репозиторий не попадает.

Зачем отдельный мастер, а не «поправь json руками»: там ровно четыре поля,
в которых легко ошибиться незаметно. Опечатка в ключе Reality роняет
sing-box на старте с `illegal base64 data at input byte 12`, и дальше по
цепочке отваливается всё — туннель, Discord, бот, — а выглядит это как
проблема с сетью. Проще проверить значения до записи.

Быстрее и надёжнее всего — вставить ссылку vless:// из панели целиком:
тогда все поля разбираются из неё и ошибиться негде.
"""

from __future__ import annotations

import base64
import json
import os
import re
import subprocess
import sys
from getpass import getpass
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlsplit

ROOT = Path(__file__).resolve().parent.parent
TEMPLATE = ROOT / "deploy" / "sing-box.json.example"
TARGET = Path("/etc/sing-box/config.json")

UUID_RE = re.compile(r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$")
HEX_RE = re.compile(r"^[0-9a-fA-F]*$")

if sys.stdout.isatty():
    OK, WARN, ERR, HEAD, OFF = "\033[32m", "\033[33m", "\033[31m", "\033[1;36m", "\033[0m"
else:
    OK = WARN = ERR = HEAD = OFF = ""


def ok(text: str) -> None:
    print(f"    {OK}[ok]{OFF}   {text}")


def warn(text: str) -> None:
    print(f"    {WARN}[warn]{OFF} {text}")


def die(text: str) -> None:
    print(f"\n{ERR}[fail]{OFF} {text}\n", file=sys.stderr)
    raise SystemExit(1)


def read(prompt: str) -> str:
    """input(), не падающий на кривой кодировке.

    Под sudo локаль часто C, а терминал шлёт что-нибудь своё: одно нажатие
    «Y» в русской раскладке — и обычный input() валится с UnicodeDecodeError
    посреди диалога, теряя всё уже введённое.
    """
    while True:
        try:
            return input(prompt).strip()
        except UnicodeDecodeError:
            print(f"    {WARN}не разобрал ввод (кодировка) — повторите{OFF}")


def yes(prompt: str) -> bool:
    """Согласие по умолчанию. Понимает и русскую раскладку."""
    return read(prompt).lower() not in ("n", "no", "н", "нет", "т")


def mask(value: str, keep: int = 4) -> str:
    if len(value) <= keep * 2:
        return "*" * len(value)
    return f"{value[:keep]}…{value[-keep:]}({len(value)})"


# ------------------------------------------------------------ проверки ----


def check_uuid(value: str) -> str:
    value = value.strip()
    if not UUID_RE.match(value):
        raise ValueError("uuid должен быть вида 8-4-4-4-12 из шестнадцатеричных цифр")
    return value


def check_public_key(value: str) -> str:
    """Ключ Reality — 43 символа base64url без padding."""
    value = value.strip().strip('"').rstrip("=")
    # Панели иногда отдают обычный base64: приводим к url-варианту сами,
    # иначе sing-box споткнётся на '+' или '/'.
    value = value.replace("+", "-").replace("/", "_")

    bad = [(i, c) for i, c in enumerate(value) if not (c.isalnum() or c in "-_")]
    if bad:
        index, char = bad[0]
        raise ValueError(f"недопустимый символ {char!r} на позиции {index}")
    if len(value) != 43:
        raise ValueError(f"длина {len(value)}, а должно быть 43 символа")
    try:
        base64.urlsafe_b64decode(value + "=")
    except Exception as exc:  # noqa: BLE001 — сообщение важнее типа
        raise ValueError(f"не разбирается как base64url: {exc}") from exc
    return value


def check_short_id(value: str) -> str:
    value = value.strip()
    if not HEX_RE.match(value):
        raise ValueError("short_id — только шестнадцатеричные цифры")
    if len(value) > 16 or len(value) % 2:
        raise ValueError("short_id — чётное число цифр, не больше 16")
    return value


def check_port(value: str) -> int:
    try:
        port = int(str(value).strip())
    except ValueError as exc:
        raise ValueError("порт — число") from exc
    if not 1 <= port <= 65535:
        raise ValueError("порт вне диапазона 1..65535")
    return port


def check_host(value: str) -> str:
    value = value.strip()
    if not value or " " in value:
        raise ValueError("адрес пустой или с пробелом")
    return value


def check_sni(value: str) -> str:
    value = value.strip()
    if not value:
        raise ValueError("SNI не может быть пустым")
    return value


# ---------------------------------------------------------------- ввод ----


def ask(prompt: str, current: str | None, validator, secret: bool = False):  # noqa: ANN001
    """Спросить значение, показывая текущее как вариант по умолчанию."""
    suffix = f" [{mask(current) if secret else current}]" if current else ""
    while True:
        if secret:
            raw = getpass(f"  {prompt}{suffix}: ").strip()
        else:
            raw = read(f"  {prompt}{suffix}: ")
        if not raw and current:
            raw = str(current)
        try:
            return validator(raw)
        except ValueError as exc:
            print(f"    {ERR}✗{OFF} {exc}")


def parse_link(link: str) -> dict:
    """Разобрать vless://uuid@host:port?...#name из панели."""
    parts = urlsplit(link.strip())
    if parts.scheme != "vless":
        raise ValueError(f"ожидалась ссылка vless://, а не {parts.scheme}://")
    if not parts.username:
        raise ValueError("в ссылке нет uuid")

    query = {key: value[0] for key, value in parse_qs(parts.query).items()}
    if query.get("security") != "reality":
        raise ValueError(f"security={query.get('security')!r}, а шаблон рассчитан на reality")
    if query.get("type", "tcp") != "tcp":
        raise ValueError(
            f"type={query.get('type')!r}. Транспорты вроде xhttp дают джиттер на голосе — "
            "заведите в панели отдельный inbound VLESS + TCP + REALITY"
        )

    return {
        "server": check_host(parts.hostname or ""),
        "server_port": check_port(parts.port or 443),
        "uuid": check_uuid(parts.username),
        "public_key": check_public_key(query.get("pbk", "")),
        "short_id": check_short_id(query.get("sid", "")),
        "server_name": check_sni(unquote(query.get("sni", ""))),
        "flow": query.get("flow", "xtls-rprx-vision"),
        "fingerprint": query.get("fp", "chrome"),
    }


def read_current() -> dict:
    """Достать уже настроенные значения, чтобы их можно было не вводить заново."""
    if not TARGET.exists():
        return {}
    try:
        config = json.loads(TARGET.read_text(encoding="utf-8"))
        outbound = next(o for o in config["outbounds"] if o.get("type") == "vless")
        tls = outbound.get("tls", {})
        reality = tls.get("reality", {})
        return {
            "server": outbound.get("server", ""),
            "server_port": outbound.get("server_port", 443),
            "uuid": outbound.get("uuid", ""),
            "public_key": reality.get("public_key", ""),
            "short_id": reality.get("short_id", ""),
            "server_name": tls.get("server_name", ""),
            "flow": outbound.get("flow", "xtls-rprx-vision"),
            "fingerprint": tls.get("utls", {}).get("fingerprint", "chrome"),
        }
    except Exception:  # noqa: BLE001 — битый конфиг это как раз наш случай
        warn("текущий конфиг не разбирается — заполняем с нуля")
        return {}


def collect() -> dict:
    current = read_current()
    if current.get("server"):
        ok(f"нашёл текущие настройки: {current['server']}:{current['server_port']}")

    print(f"\n{HEAD}Вставьте ссылку vless:// из панели{OFF}")
    print("  (Enter — ввести поля по одному)\n")
    link = read("  ссылка: ")

    if link:
        try:
            values = parse_link(link)
        except ValueError as exc:
            die(f"ссылка не разобралась: {exc}")
        ok("ссылка разобрана, все поля заполнены")
        return values

    print(f"\n{HEAD}Параметры подключения{OFF}")
    return {
        "server": ask("адрес VPS", current.get("server"), check_host),
        "server_port": ask("порт", current.get("server_port"), check_port),
        "uuid": ask("uuid", current.get("uuid"), check_uuid, secret=True),
        "public_key": ask("public key (Reality)", current.get("public_key"), check_public_key),
        "short_id": ask("short id", current.get("short_id"), check_short_id),
        "server_name": ask("SNI (dest)", current.get("server_name"), check_sni),
        "fingerprint": ask("fingerprint", current.get("fingerprint") or "chrome", str.strip),
        "flow": current.get("flow") or "xtls-rprx-vision",
    }


# --------------------------------------------------------------- запись ---


def build_config(values: dict) -> dict:
    """Взять шаблон из репозитория и подставить в него параметры.

    Шаблон, а не сборка с нуля: inbounds, dns и route должны оставаться
    такими же, как в git, иначе правки шаблона не будут доезжать до машин.
    """
    if not TEMPLATE.exists():
        die(f"нет шаблона {TEMPLATE}")

    config = json.loads(TEMPLATE.read_text(encoding="utf-8"))
    config.pop("_комментарий", None)

    for outbound in config["outbounds"]:
        if outbound.get("type") != "vless":
            continue
        outbound["server"] = values["server"]
        outbound["server_port"] = values["server_port"]
        outbound["uuid"] = values["uuid"]
        outbound["flow"] = values["flow"]
        tls = outbound.setdefault("tls", {})
        tls["server_name"] = values["server_name"]
        tls.setdefault("utls", {})["fingerprint"] = values["fingerprint"]
        reality = tls.setdefault("reality", {})
        reality["public_key"] = values["public_key"]
        reality["short_id"] = values["short_id"]
        break
    else:
        die("в шаблоне нет outbound типа vless")

    return config


def write(config: dict) -> None:
    TARGET.parent.mkdir(parents=True, exist_ok=True)
    # 0600 до записи, а не после: иначе ключи успевают полежать доступными.
    handle = os.open(TARGET, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(handle, "w", encoding="utf-8") as fh:
        json.dump(config, fh, ensure_ascii=False, indent=2)
        fh.write("\n")
    ok(f"{TARGET} записан (права 0600)")


def verify_and_restart() -> None:
    binary = "/usr/local/bin/sing-box"
    if not Path(binary).exists():
        warn(f"{binary} не найден — сначала sudo ./scripts/install_singbox.sh")
        return

    check = subprocess.run(
        [binary, "check", "-c", str(TARGET)], capture_output=True, text=True
    )
    if check.returncode != 0:
        die(f"sing-box не принял конфиг:\n       {check.stderr.strip() or check.stdout.strip()}")
    ok("конфиг валиден")

    if not yes("\n  Перезапустить sing-box и проверить туннель? [Y/n]: "):
        print("\n  Тогда вручную: sudo systemctl restart sing-box\n")
        return

    subprocess.run(["systemctl", "enable", "--now", "sing-box"], check=False)
    subprocess.run(["systemctl", "restart", "sing-box"], check=False)

    # sing-box при старте прописывает себя резолвером всего хоста — снимаем.
    subprocess.run(
        ["resolvectl", "revert", os.environ.get("BASHMAK_TUN_IF", "tun-bashmak")],
        check=False,
        capture_output=True,
    )

    probe = subprocess.run(
        [
            "curl", "-sS", "--socks5-hostname", "127.0.0.1:10808", "--max-time", "15",
            "-o", "/dev/null", "-w", "%{http_code}",
            "https://discord.com/api/v10/gateway",
        ],
        capture_output=True,
        text=True,
    )
    if probe.stdout.strip() == "200":
        ok("туннель работает: Discord отвечает 200 через SOCKS")
        print("\n  Дальше: ./scripts/setup.sh --tunnel && ./scripts/doctor.sh\n")
    else:
        warn(f"через туннель код {probe.stdout.strip() or '—'} {probe.stderr.strip()}")
        warn("смотрите: journalctl -u sing-box -n 30 --no-pager")


def main() -> None:
    if os.geteuid() != 0:
        die(f"нужны права root: sudo {Path(sys.argv[0]).as_posix()}")

    print(f"\n{HEAD}Башмак — настройка туннеля sing-box{OFF}")

    values = collect()

    print(f"\n{HEAD}Проверьте{OFF}")
    print(f"  адрес:       {values['server']}:{values['server_port']}")
    print(f"  uuid:        {mask(values['uuid'])}")
    print(f"  public key:  {mask(values['public_key'])}")
    print(f"  short id:    {values['short_id']}")
    print(f"  SNI:         {values['server_name']}")
    print(f"  flow / fp:   {values['flow']} / {values['fingerprint']}")

    if not yes("\n  Записать? [Y/n]: "):
        print("\n  Отменено, ничего не тронуто.\n")
        return

    write(build_config(values))
    verify_and_restart()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n  Прервано, файл не тронут.\n")
        raise SystemExit(130) from None
