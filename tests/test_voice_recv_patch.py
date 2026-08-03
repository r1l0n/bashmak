"""Заплатки на приём голоса: перескок дыры в буфере и расшифровка DAVE.

Настоящие ``HeapJitterBuffer`` и ``DaveSession`` тянут за собой discord.py и
davey, поэтому здесь заглушки с тем же контрактом. Ловим два главных свойства:
пакеты уходят по возрастанию номера и ни один не теряется; кадр расшифровывают
ровно тогда, когда это можно, и не трогают, когда нельзя.
"""

from __future__ import annotations

import heapq
from dataclasses import dataclass, field

from bashmak.audio.voice_recv_patch import dave_unwrap, take_lowest


@dataclass(order=True)
class _Packet:
    sequence: int


@dataclass
class _Buffer:
    _buffer: list = field(default_factory=list)
    _last_tx_seq: int = -1
    updates: int = 0

    def push(self, sequence: int) -> None:
        heapq.heappush(self._buffer, _Packet(sequence))

    def _update_has_item(self) -> None:
        self.updates += 1


def test_отдаёт_пакеты_по_возрастанию_и_ничего_не_теряет():
    buffer = _Buffer()
    # Порядок прихода перепутан, как оно и бывает на туннеле.
    for sequence in (12, 10, 14, 11):
        buffer.push(sequence)

    drained = []
    while (packet := take_lowest(buffer)) is not None:
        drained.append(packet.sequence)

    assert drained == [10, 11, 12, 14]
    assert buffer._last_tx_seq == 14
    assert buffer.updates == len(drained)


def test_пустой_буфер_не_падает():
    assert take_lowest(_Buffer()) is None


# ------------------------------------------------------ расшифровка DAVE ----

AUDIO = object()  # вместо davey.MediaType.audio
CIPHER = b"\x01\x02\x03"
PLAIN = b"opus"


@dataclass
class _Session:
    ready: bool = True
    passthrough: set = field(default_factory=set)
    calls: list = field(default_factory=list)

    def can_passthrough(self, user_id: int) -> bool:
        return user_id in self.passthrough

    def decrypt(self, user_id: int, media_type, payload: bytes) -> bytes:
        self.calls.append((user_id, media_type, payload))
        return PLAIN


def test_расшифровывает_когда_сессия_готова():
    session = _Session()
    assert dave_unwrap(session, 1, 4711, AUDIO, CIPHER) == PLAIN
    assert session.calls == [(4711, AUDIO, CIPHER)]


def test_не_трогает_кадр_когда_расшифровывать_нельзя():
    # Каждый случай отдельно: любой из них, тронутый по ошибке, кончается
    # исключением из davey на каждом кадре речи.
    cases = {
        "нет сессии": (None, 1, 4711),
        "сессия не готова": (_Session(ready=False), 1, 4711),
        "канал без E2EE": (_Session(), 0, 4711),
        "SSRC не сопоставлен": (_Session(), 1, None),
        "участник в passthrough": (_Session(passthrough={4711}), 1, 4711),
    }
    for name, (session, version, user_id) in cases.items():
        assert dave_unwrap(session, version, user_id, AUDIO, CIPHER) is CIPHER, name
        if session is not None:
            assert session.calls == [], name


def test_пустой_кадр_проходит_насквозь():
    # FakePacket, которым буфер затыкает дыру.
    session = _Session()
    assert dave_unwrap(session, 1, 4711, AUDIO, b"") == b""
    assert session.calls == []
