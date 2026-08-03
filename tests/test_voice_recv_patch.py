"""Проверка перескока дыры в джиттер-буфере.

Настоящий ``JitterBuffer`` тянет за собой py-cord, поэтому здесь заглушка с
тем же контрактом: куча пакетов в ``_buffer``, последний отданный номер в
``_last_tx_seq``. Ловим главное — что пакеты уходят по возрастанию номера и
что ни один не теряется.
"""

from __future__ import annotations

import heapq
from dataclasses import dataclass, field

from bashmak.audio.pycord_patch import take_lowest


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
