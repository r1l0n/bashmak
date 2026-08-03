"""Заплатки на приёмный тракт py-cord 2.8.1.

Приём голоса в 2.8.1 не работает ни в каком виде — это признаёт сама
библиотека (``RuntimeWarning`` при ``start_recording`` и issue
`#3139 <https://github.com/Pycord-Development/pycord/issues/3139>`_).
Здесь собрано всё, что нужно поправить, чтобы из канала пошёл чистый PCM.
Патчи ставятся на классы **до** первого подключения к голосу: декриптор
разбирает методы по имени режима в ``__init__``, и после создания объекта
подменять уже поздно.

Почему нельзя просто отказаться от E2EE
---------------------------------------
Первой мыслью было объявить Discord'у нулевую версию DAVE и жить на
транспортном шифровании. Не выйдет: канал отвечает close code **4017**,
документированный как *«E2EE/DAVE protocol required: this channel requires
a client supporting E2EE via the DAVE Protocol»*, и голосовой шлюз рвёт
соединение ещё до приёма. Тем же кодом, к слову, объясняются старые падения
на py-cord 2.6.1 и 2.7.2 — они про DAVE не знают вовсе.

Значит DAVE должен остаться включённым, и чинить надо расшифровку.

Что именно чинится
------------------

1. **RTP-расширение срезается по фактической длине.**
   ``_decrypt_rtp_aead_xchacha20_poly1305_rtpsize`` считает смещение через
   ``update_extended_header()``, выбрасывает его и режет захардкоженные
   8 байт. Это верно только когда расширение занимает ровно два слова.

2. **Расширение срезается ровно один раз.** Оригинальная ``decrypt_rtp()``
   режет его повторно — уже из данных, расшифрованных DAVE, — и уносит
   начало Opus. Отсюда ``OpusError('corrupted stream')`` на первом же пакете
   речи. DAVE шифрует только полезную нагрузку, расширение к этому моменту
   давно снято.

3. **``decrypt_rtp()`` всегда возвращает расшифрованное.** В оригинале
   результат кладётся в локальную переменную, а наружу отдаётся
   ``packet.decrypted_data``, которое присваивается только внутри ветки
   DAVE. Если SSRC ещё не сопоставлен с человеком, оттуда уходит ``None`` —
   и ``AudioReader.callback`` молча выбрасывает пакет.

4. **Бессмысленная расшифровка поверх готового PCM убрана.** В ``opus.py``
   после декодирования остался вызов ``dave.decrypt()`` уже по декодированным
   сэмплам. Флаг ``HAS_DAVEY`` читается в этом файле ровно в одном месте —
   в условии перед этим вызовом, — поэтому достаточно погасить его копию в
   модуле ``discord.opus``, не трогая остальную библиотеку.

5. **Ошибка декодирования перестаёт убивать приём.** Исключение из
   ``PacketDecoder.pop_data()`` выходит в ``PacketRouter._do_run``, поток
   роутера умирает и в ``finally`` зовёт ``stop_recording()`` — бот остаётся
   в канале навсегда глухим. Битый пакет должен стоить одного пакета.

Пункты 1, 2 и 5 совпадают по смыслу с форком
`vito1317/pycord@5a95f98 <https://github.com/vito1317/pycord/commit/5a95f984>`_
— диагноз про двойное срезание расширения найден там. Сам форк не
используется: он ответвлён от master до выхода 2.8.0, а подменять библиотеку
целиком в процессе, который держит токен, — лишний риск.
"""

from __future__ import annotations

import heapq
import logging

log = logging.getLogger(__name__)

#: Сколько пропущенных пакетов между сообщениями в лог.
_DROP_LOG_EVERY = 500


def apply() -> None:
    """Поставить все заплатки. Зовётся один раз на старте процесса."""
    _patch_rtp_decryption()
    _drop_dave_decrypt_over_pcm()
    _patch_gap_recovery()
    _patch_decoder_resilience()


def _passthrough(dave, user_id: int) -> bool:  # noqa: ANN001 — davey.DaveSession
    """Шлёт ли этот участник открытые кадры.

    Сессия DAVE стартует именно в этом режиме (``set_passthrough_mode`` в
    ``voice/state.py``), шифрование включается позже. Расшифровывать открытый
    кадр нельзя — davey ответит ``NoValidCryptorFound``.
    """
    check = getattr(dave, "can_passthrough", None)
    return bool(check and check(user_id))


def take_lowest(buffer) -> object | None:  # noqa: ANN001 — discord JitterBuffer
    """Снять пакет с наименьшим ``sequence``, не трогая остальные.

    Замена штатному «выбросить буфер целиком». Лезет во внутренности
    ``JitterBuffer`` осознанно: публичного способа достать один пакет в обход
    проверки на дыру в последовательности у него нет.
    """
    heap = buffer._buffer
    if not heap:
        return None

    packet = heapq.heappop(heap)
    buffer._last_tx_seq = packet.sequence
    buffer._update_has_item()
    return packet


def _patch_rtp_decryption() -> None:
    """Пункты 1–3: расшифровка RTP до чистого Opus."""
    from discord.voice.receive import reader as voice_reader

    decryptor = getattr(voice_reader, "PacketDecryptor", None)
    if decryptor is None:
        log.warning("не нашёл PacketDecryptor — приём голоса работать не будет")
        return

    crypto_error = getattr(voice_reader, "CryptoError", Exception)
    davey = getattr(voice_reader, "davey", None)
    if davey is None:
        log.error("нет пакета davey — канал с обязательным E2EE не примет бота")
        return

    unknown_ssrcs: set[int] = set()
    failures = 0

    def decrypt_rtp(self, packet):  # noqa: ANN001 — сигнатура задана py-cord
        """Транспорт → снятие расширения → DAVE. Расширение режется один раз."""
        payload = self._decryptor_rtp(packet)
        state = self.client._connection
        dave = getattr(state, "dave_session", None)

        if dave is not None and dave.ready:
            user_id = state.ssrc_user_map.get(packet.ssrc)
            if not user_id:
                # Мапа SSRC→человек ещё не приехала. Расшифровать нечем, а
                # отдавать шифротекст в Opus незачем — пропускаем пакет.
                if packet.ssrc not in unknown_ssrcs:
                    unknown_ssrcs.add(packet.ssrc)
                    log.debug("SSRC %s ещё не сопоставлен — пакеты пропускаются", packet.ssrc)
                packet.decrypted_data = b""
                return b""

            unknown_ssrcs.discard(packet.ssrc)
            if not _passthrough(dave, user_id):
                try:
                    payload = dave.decrypt(user_id, davey.MediaType.audio, payload)
                except Exception as exc:
                    # payload не трогаем: кадр мог оказаться открытым. Если это
                    # всё же шифротекст, его отбракует Opus, и пакет пропадёт
                    # один, а не подменится тишиной на весь разговор.
                    nonlocal failures
                    failures += 1
                    if failures == 1 or failures % _DROP_LOG_EVERY == 0:
                        log.warning("DAVE не расшифровал пакет (%d-й): %s", failures, exc)

        packet.decrypted_data = payload
        return payload

    def decrypt_xchacha(self, packet):  # noqa: ANN001 — сигнатура задана py-cord
        """То же, что в 2.8.1, но расширение режется по фактической длине."""
        packet.adjust_rtpsize()
        nonce = packet.nonce + b"\x00" * 20

        try:
            result = self.box.decrypt(
                packet.decrypted_data or packet.data,
                bytes(packet.header),
                nonce,
            )
        except Exception as exc:
            raise crypto_error(exc)

        if packet.extended:
            # update_extended_header() и разбирает расширение, и возвращает
            # смещение до полезной нагрузки. Оригинал возвращал result[8:].
            return result[packet.update_extended_header(result) :]
        return result

    decryptor.decrypt_rtp = decrypt_rtp
    decryptor._decrypt_rtp_aead_xchacha20_poly1305_rtpsize = decrypt_xchacha


def _drop_dave_decrypt_over_pcm() -> None:
    """Пункт 4: погасить HAS_DAVEY в discord.opus.

    Единственное, что этот флаг охраняет в файле, — вызов ``dave.decrypt()``
    по уже декодированным сэмплам. Копия флага в
    ``discord.voice.utils.dependencies`` не трогается, DAVE остаётся живым.
    """
    from discord import opus as discord_opus

    if not getattr(discord_opus, "HAS_DAVEY", False):
        return
    discord_opus.HAS_DAVEY = False


def _patch_gap_recovery() -> None:
    """Пункт 6: потерянный пакет не должен уносить с собой весь буфер.

    ``_flag_ready_state`` будит роутер по ``peek()`` — то есть как только в
    джиттер-буфере накопилось больше ``pref_size`` пакетов. А ``pop()``
    отдаёт пакет только когда взведён ``_has_item``, то есть когда
    последовательность идёт без дыр. Условия разные, и на каждой дыре
    ``pop()`` возвращает ``None``, после чего оригинальный
    ``_get_next_packet`` делает ``flush()`` и оставляет один пакет из
    десяти — «N packets were lost being flushed» в логе. На туннеле, где
    потери обычное дело, до VAD доходят обрывки, и речи в них нет.

    Вместо этого перескакиваем дыру: берём следующий по порядку пакет,
    остальные остаются в буфере.
    """
    from discord import opus as discord_opus

    decoder = getattr(discord_opus, "PacketDecoder", None)
    if decoder is None:
        log.warning("не нашёл PacketDecoder — потери пакетов будут съедать речь")
        return

    def _get_next_packet(self, timeout: float):
        packet = self._buffer.pop(timeout=timeout)
        if packet is None:
            packet = take_lowest(self._buffer)
            if packet is None:
                return None
        if not packet:
            packet = self._make_fakepacket()
        return packet

    decoder._get_next_packet = _get_next_packet


def _patch_decoder_resilience() -> None:
    """Пункт 5: битый пакет стоит пакета, а не всей сессии приёма."""
    from discord import opus as discord_opus

    decoder = getattr(discord_opus, "PacketDecoder", None)
    if decoder is None:
        log.warning("не нашёл PacketDecoder — приём не переживёт битый пакет")
        return

    original = decoder.pop_data
    opus_error = discord_opus.OpusError
    dropped = 0

    def pop_data(self, *, timeout: float = 0):
        nonlocal dropped
        try:
            return original(self, timeout=timeout)
        except opus_error as exc:
            dropped += 1
            if dropped == 1 or dropped % _DROP_LOG_EVERY == 0:
                log.warning("пропущен неразобранный голосовой пакет (%d-й): %s", dropped, exc)
            # None роутер просто пропустит, не трогая синк.
            return None

    decoder.pop_data = pop_data
