"""Кастомный AudioSink py-cord: разводка входящего голоса по говорящим.

py-cord отдаёт голос уже раздельно по SSRC — смешивать ничего не нужно,
достаточно разложить кадры по буферам и не тормозить.

``write()`` вызывается из потока приёма голоса, общего на весь голосовой
клиент. Всё, что здесь делается, — микс стерео в моно, ресемпл 48→16 кГц и
append в дек. VAD, STT и прочее живут в asyncio-лупе (см. listener.py).

Про совместимость с py-cord 2.8.x
---------------------------------
В 2.8 приёмный стек переехал в ``discord/voice/receive/`` и обзавёлся
событийным роутером, но на старый ``discord.sinks.Sink`` его не перевели:
новый код требует от синка ``__sink_listeners__``, ``walk_children()`` и
``is_opus()``, а на базовом классе их нет — падает даже штатный
``WaveSink``. Поэтому мы объявляем их сами:

* ``__sink_listeners__`` пустой — звук приходит в ``write()``
  (``router.py``: ``self.sink.write(data, data.source)``), а события
  (speaking start/stop, rtcp) нам не нужны;
* ``walk_children()`` пустой — вложенных синков у нас нет;
* ``is_opus()`` → ``False``, чтобы ``PacketDecoder`` сам раскодировал Opus
  и клал в ``VoiceData.pcm`` обычный PCM 48 кГц / 2 канала / s16le.

Второе, чего 2.8.1 не делает сама: ``start_recording()`` больше не зовёт
``Sink.init(vc)``, а ``PacketDecoder._process_packet`` начинается с
``assert self.sink.client`` и лезет в ``client._ssrc_to_id`` за автором
пакета. Без ``vc`` приём падает на первом же кадре — поэтому его
проставляют явно (см. ``GuildSession.start()`` в ``bashmak/bot.py``).

Ниже 2.8 идти нельзя, даже ради старого API: Discord рвёт голосовое
соединение с кодом 4017 — версии до 2.8 говорят на протоколе, который
сервер больше не принимает.
"""

from __future__ import annotations

import logging
from typing import Iterator

from discord.sinks import Sink

from .buffer import StreamRegistry
from .resample import DISCORD_RATE, WORK_RATE, StreamResampler, discord_pcm_to_mono

log = logging.getLogger(__name__)


def _extract_pcm(data) -> bytes | None:  # noqa: ANN001 — тип зависит от версии py-cord
    """Достать сырой PCM.

    До 2.8 в ``write()`` приходили байты, в 2.8 — ``VoiceData`` с полями
    ``pcm``, ``source`` и ``packet``. Поддерживаем оба, чтобы обновление
    библиотеки не роняло приём молча: тут поток чужой, и исключение отсюда
    глушит голос всем сразу.
    """
    if isinstance(data, (bytes, bytearray, memoryview)):
        return bytes(data)
    for attribute in ("pcm", "decoded_data", "data", "audio"):
        payload = getattr(data, attribute, None)
        if isinstance(payload, (bytes, bytearray, memoryview)):
            return bytes(payload)
    return None


def _extract_user_id(user, data) -> int | None:  # noqa: ANN001
    """Кто говорит. Может прийти id, объект Member или ничего."""
    for candidate in (user, getattr(data, "source", None), getattr(data, "user_id", None)):
        if candidate is None:
            continue
        identifier = getattr(candidate, "id", candidate)
        if isinstance(identifier, int):
            return identifier
    return None


class BashmakSink(Sink):
    """Sink, который ничего не пишет на диск — только раздаёт PCM пайплайну."""

    #: Событийный роутер 2.8 читает это у синка. Речь приходит через write(),
    #: события (speaking start/stop, rtcp) нам не нужны — список пустой.
    __sink_listeners__: tuple[tuple[str, str], ...] = ()

    def __init__(self, registry: StreamRegistry) -> None:
        super().__init__()
        self.registry = registry
        self._resamplers: dict[int, StreamResampler] = {}
        self._warned: set[int] = set()
        # client трогать не надо: в 2.8 это property только на чтение,
        # присваивание падает с AttributeError. Роутер читает его сам.

    def walk_children(self) -> Iterator[Sink]:
        """Синк один, вложенных нет. Роутер обходит дерево — отдаём пустое."""
        return iter(())

    def is_opus(self) -> bool:
        """Нет: пусть PacketDecoder декодирует Opus, нам нужен готовый PCM."""
        return False

    # py-cord зовёт это на каждые 20 мс речи каждого участника.
    def write(self, data, user=None) -> None:  # noqa: ANN001 — сигнатура задана py-cord
        user_id = None
        try:
            user_id = _extract_user_id(user, data)
            if user_id is None:
                return

            pcm = _extract_pcm(data)
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
            # Исключение здесь убило бы поток приёма для ВСЕХ участников,
            # поэтому логируем один раз на пользователя и продолжаем.
            if user_id not in self._warned:
                self._warned.add(user_id)
                log.exception("сбой обработки входящего кадра, user=%s", user_id)

    def forget(self, user_id: int) -> None:
        """Забыть состояние ушедшего участника."""
        self._resamplers.pop(user_id, None)
        self._warned.discard(user_id)
        self.registry.drop(user_id)

    # Базовый Sink на этом этапе склеивает wav-файлы — нам это не нужно.
    def cleanup(self) -> None:
        self.finished = True
        self._resamplers.clear()
        self.registry.clear()

    def format_audio(self, audio) -> None:  # noqa: ANN001 — сигнатура задана py-cord
        return None
