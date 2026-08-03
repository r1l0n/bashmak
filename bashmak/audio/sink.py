"""Кастомный AudioSink: разводка входящего голоса по говорящим.

discord-ext-voice-recv отдаёт голос уже раздельно по SSRC — смешивать ничего
не нужно, достаточно разложить кадры по буферам и не тормозить.

``write()`` вызывается из потока роутера пакетов, общего на весь голосовой
клиент. Всё, что здесь делается, — микс стерео в моно, ресемпл 48→16 кГц и
append в дек. VAD, STT и прочее живут в asyncio-лупе (см. listener.py).

``wants_opus() → False`` означает, что библиотека сама раскодирует Opus и
положит в ``data.pcm`` **48 кГц / 2 канала / s16le** — ровно то, что ждёт
``discord_pcm_to_mono()``.

Чего библиотека не делает сама — снимает E2EE с кадра. Расшифровку DAVE
ставит :mod:`bashmak.audio.voice_recv_patch` до подключения к каналу; без неё
в ``pcm`` приезжает шум.
"""

from __future__ import annotations

import logging

from discord.ext.voice_recv import AudioSink

from .buffer import StreamRegistry
from .resample import DISCORD_RATE, WORK_RATE, StreamResampler, discord_pcm_to_mono

log = logging.getLogger(__name__)


class BashmakSink(AudioSink):
    """Sink, который ничего не пишет на диск — только раздаёт PCM пайплайну."""

    def __init__(self, registry: StreamRegistry) -> None:
        super().__init__()
        self.registry = registry
        self._resamplers: dict[int, StreamResampler] = {}
        self._warned: set[int | None] = set()

    def wants_opus(self) -> bool:
        """Нет: пусть PacketDecoder декодирует Opus, нам нужен готовый PCM."""
        return False

    def _user_id(self, user, data) -> int | None:  # noqa: ANN001 — типы заданы библиотекой
        """Кто говорит.

        ``user`` — ``Member``/``User`` или ``None``: библиотека ищет человека
        в кэше гильдии, и до первого события speaking (или если участника нет
        в кэше) там пусто. Тогда спрашиваем голосовой клиент напрямую — мапу
        SSRC→id он ведёт сам.
        """
        if user is not None:
            return user.id

        client = self.voice_client
        ssrc = getattr(getattr(data, "packet", None), "ssrc", None)
        if client is None or ssrc is None:
            return None
        return client._get_id_from_ssrc(ssrc)

    # Библиотека зовёт это на каждые 20 мс речи каждого участника.
    def write(self, user, data) -> None:  # noqa: ANN001 — сигнатура задана библиотекой
        user_id = None
        try:
            user_id = self._user_id(user, data)
            if user_id is None:
                return

            pcm = data.pcm
            if not pcm:
                return

            mono48 = discord_pcm_to_mono(pcm)
            if mono48.size == 0:
                return

            resampler = self._resamplers.get(user_id)
            if resampler is None:
                resampler = StreamResampler(DISCORD_RATE, WORK_RATE)
                self._resamplers[user_id] = resampler

            mono16 = resampler.process(mono48)
            if mono16.size:
                self.registry.get(user_id).push(mono16)
        except Exception:
            # Исключение здесь убило бы поток роутера, а с ним приём для ВСЕХ
            # участников, поэтому логируем один раз на пользователя и живём
            # дальше.
            if user_id not in self._warned:
                self._warned.add(user_id)
                log.exception("сбой обработки входящего кадра, user=%s", user_id)

    def forget(self, user_id: int) -> None:
        """Забыть состояние ушедшего участника."""
        self._resamplers.pop(user_id, None)
        self._warned.discard(user_id)
        self.registry.drop(user_id)

    def cleanup(self) -> None:
        # Зовётся и из AudioReader._stop(), и из AudioSink.__del__ — должен
        # переживать повторный вызов.
        self._resamplers.clear()
        self.registry.clear()
