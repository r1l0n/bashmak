"""Загрузка config.yaml и .env.

Все относительные пути в конфиге считаются от корня проекта, а не от текущего
рабочего каталога: иначе бот, запущенный из systemd, искал бы модели не там.

Значения по умолчанию берутся из config.example.yaml, а config.yaml
накладывается поверх. Так добавленный в шаблон ключ доезжает и до тех, у кого
config.yaml создан давно (setup.sh существующий файл не трогает), а ключ,
которого в шаблоне нет, виден как опечатка.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent
#: Шаблон конфига: он же перечень всех известных ключей и их значений.
EXAMPLE = ROOT / "config.example.yaml"


class Section:
    """Секция конфига с доступом через точку: cfg.vad.threshold."""

    def __init__(self, data: dict[str, Any], root: Path) -> None:
        self._data = data
        self._root = root

    def __getattr__(self, name: str) -> Any:
        try:
            value = self._data[name]
        except KeyError:
            raise AttributeError(
                f"в конфиге нет ключа '{name}' (есть: {', '.join(sorted(self._data))})"
            ) from None
        return Section(value, self._root) if isinstance(value, dict) else value

    def __contains__(self, name: str) -> bool:
        return name in self._data

    def get(self, name: str, default: Any = None) -> Any:
        value = self._data.get(name, default)
        return Section(value, self._root) if isinstance(value, dict) else value

    def path(self, name: str) -> Path:
        """Значение ключа как абсолютный путь (относительные — от корня проекта).

        Через __getattr__, а не self._data[name]: голый KeyError не говорит, в
        какой секции искать, а AttributeError отсюда перечисляет соседние ключи.
        """
        raw = Path(str(getattr(self, name))).expanduser()
        return raw if raw.is_absolute() else (self._root / raw)

    def as_dict(self) -> dict[str, Any]:
        return self._data

    def __repr__(self) -> str:  # pragma: no cover — только для отладки
        return f"Section({self._data!r})"


class Config(Section):
    """Корень конфига + доступ к секретам из окружения."""

    def __init__(self, data: dict[str, Any], root: Path, source: Path) -> None:
        super().__init__(data, root)
        self.root = root
        self.source = source

    @property
    def discord_token(self) -> str:
        token = os.environ.get("DISCORD_TOKEN", "").strip()
        if not token:
            raise RuntimeError(
                "DISCORD_TOKEN не задан. Впишите его в .env "
                f"(шаблон — {self.root / '.env.example'})"
            )
        return token


def merge_defaults(
    defaults: dict[str, Any], override: dict[str, Any], _path: str = ""
) -> dict[str, Any]:
    """Наложить пользовательский конфиг на шаблон, секция за секцией.

    Ключ, которого нет в шаблоне, — почти всегда опечатка. Пишем о нём в лог, но
    не падаем: лишняя строка в конфиге не повод не поднимать бота.
    """
    merged = dict(defaults)
    for key, value in override.items():
        if key not in defaults:
            log.warning(
                "config.yaml: ключ %r ничего не настраивает — опечатка или "
                "настройка, которой больше нет (сверьтесь с config.example.yaml)",
                f"{_path}{key}",
            )
        base = defaults.get(key)
        if isinstance(base, dict) and isinstance(value, dict):
            merged[key] = merge_defaults(base, value, f"{_path}{key}.")
        else:
            merged[key] = value
    return merged


def _read_yaml(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def load_config(path: str | os.PathLike[str] | None = None) -> Config:
    """Прочитать конфиг. Порядок: аргумент → $BASHMAK_CONFIG → <root>/config.yaml."""
    load_dotenv(ROOT / ".env")

    candidate = Path(path) if path else Path(os.environ.get("BASHMAK_CONFIG", ROOT / "config.yaml"))
    if not candidate.is_absolute():
        candidate = ROOT / candidate

    if not candidate.exists():
        raise FileNotFoundError(
            f"нет файла конфигурации {candidate}. Запустите ./scripts/setup.sh "
            "или скопируйте config.example.yaml в config.yaml"
        )

    data = _read_yaml(candidate)
    if EXAMPLE.exists():
        data = merge_defaults(_read_yaml(EXAMPLE), data)

    return Config(data, ROOT, candidate)
