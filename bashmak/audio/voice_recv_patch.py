"""Заплатки на приёмный тракт discord-ext-voice-recv.

Библиотеку не трогали с июня 2025 года, и про DAVE она не знает вовсе. А с
2 марта 2026 Discord требует E2EE от всех: без поддержки DAVE канал отвечает
close code **4017** и рвёт соединение, а после транспортной расшифровки кадр
остаётся MLS-шифротекстом. Opus разбирает у такого кадра только открытый
заголовок — отсюда «corrupted stream» вперемешку с роботизированным звуком
(issue `#53 <https://github.com/imayhaveborkedit/discord-ext-voice-recv/issues/53>`_
и `#60 <https://github.com/imayhaveborkedit/discord-ext-voice-recv/issues/60>`_
апстрима; в #60 это объясняют арифметикой RTP-расширений, но смещения там
как раз верные — дело именно в нерасшифрованном DAVE).

Сам MLS-обмен делает discord.py 2.7: ключевые пакеты, welcome, commit,
транзишены и даунгрейды живут в ``DiscordVoiceWebSocket.received_binary_message``,
а готовая сессия лежит в ``vc._connection.dave_session``. Нам остаётся позвать
у неё ``decrypt()`` на приёме — это самая дешёвая часть работы, и она здесь.

Что чинится
-----------

1. **Расшифровка DAVE перед декодированием Opus.** По смыслу совпадает с
   `PR #58 <https://github.com/imayhaveborkedit/discord-ext-voice-recv/pull/58>`_
   апстрима, плюс проверка ``can_passthrough()``: участник, ещё не перешедший
   на шифрование, шлёт открытые кадры, и davey отвечает на них
   ``NoValidCryptorFound``.

2. **Потерянный пакет не уносит с собой весь джиттер-буфер.**

3. **Битый пакет стоит пакета, а не всей сессии приёма.**

Пункты 2 и 3 дословно повторяют заплатки, которые до этого стояли здесь на
py-cord 2.8.1: её приёмный стек — перенос этой самой библиотеки, и оба бага
уехали вместе с кодом. В апстриме это PR #57 и issue #49/#51, все открытые.

Заплатки ставятся на классы, поэтому объекты, созданные позже, их подхватят.
Всё равно зовём ``apply()`` в ``main()`` до первого подключения: так порядок
не зависит от того, когда роутер решит завести декодер под новый SSRC.
"""

from __future__ import annotations

import heapq
import logging

log = logging.getLogger(__name__)

#: Сколько пропущенных кадров между сообщениями в лог.
_DROP_LOG_EVERY = 500


def apply() -> None:
    """Поставить все заплатки. Зовётся один раз на старте процесса."""
    _patch_dave_decrypt()
    _patch_gap_recovery()
    _patch_decoder_resilience()


def _passthrough(session, user_id: int) -> bool:  # noqa: ANN001 — davey.DaveSession
    """Шлёт ли этот участник открытые кадры.

    Сессия DAVE стартует именно в этом режиме, шифрование включается позже.
    Расшифровывать открытый кадр нельзя — davey ответит ``NoValidCryptorFound``.
    """
    check = getattr(session, "can_passthrough", None)
    return bool(check and check(user_id))


def dave_unwrap(session, protocol_version: int, user_id: int | None, media_type, payload: bytes) -> bytes:  # noqa: ANN001
    """Снять E2EE с кадра. Отдаёт ``payload`` нетронутым, если снимать нечего.

    Отдельной функцией, а не куском ``_process_packet``, потому что здесь
    четыре условия, при каждом из которых расшифровывать **нельзя**, и цена
    ошибки в любом — либо исключение из davey на каждом кадре, либо немой
    канал, который снаружи выглядит рабочим:

    * пустой ``payload`` — FakePacket, которым буфер затыкает дыру;
    * нет сессии или она ещё не готова;
    * версия протокола 0 — канал без E2EE или сессию понизили;
    * SSRC ещё не сопоставлен с человеком: ключ у DAVE свой на каждого
      отправителя, без id расшифровывать нечем;
    * участник в passthrough — он шлёт открытые кадры, и davey ответит на них
      ``NoValidCryptorFound``.

    Проверяется в ``tests/test_voice_recv_patch.py``.
    """
    if not payload:
        return payload
    if session is None or not session.ready or protocol_version == 0:
        return payload
    if user_id is None:
        return payload
    if _passthrough(session, user_id):
        return payload
    return session.decrypt(user_id, media_type, bytes(payload))


def take_lowest(buffer) -> object | None:  # noqa: ANN001 — HeapJitterBuffer
    """Снять пакет с наименьшим ``sequence``, не трогая остальные.

    Замена штатному «выбросить буфер целиком». Лезет во внутренности
    ``HeapJitterBuffer`` осознанно: публичного способа достать один пакет в
    обход проверки на дыру в последовательности у него нет.
    """
    heap = buffer._buffer
    if not heap:
        return None

    packet = heapq.heappop(heap)
    buffer._last_tx_seq = packet.sequence
    buffer._update_has_item()
    return packet


def _patch_dave_decrypt() -> None:
    """Пункт 1: снять E2EE с кадра, прежде чем отдавать его в Opus."""
    from discord.ext.voice_recv import opus as recv_opus

    try:
        import davey
    except ImportError:
        log.error("нет пакета davey — канал с обязательным E2EE не примет бота")
        return

    decoder = recv_opus.PacketDecoder
    voice_data = recv_opus.VoiceData

    modes: dict[int, str] = {}
    failures = 0

    def report_mode(ssrc: int, session, version: int, user_id: int | None) -> None:  # noqa: ANN001
        """Сказать в лог, что происходит с кадрами этого SSRC.

        Только при смене режима, поэтому строк будет единицы, а не 50 в
        секунду. Без этого по логу нельзя отличить рабочий приём от «звук
        как R2D2»: в обоих случаях PCM доходит до буферов, но во втором в
        Opus уходит нерасшифрованный шифротекст.
        """
        if user_id is None:
            # Мапа SSRC→человек ещё не приехала. Кадр уйдёт в Opus как есть:
            # негодный отбракует декодер, а подменять его тишиной нельзя —
            # канал будет выглядеть рабочим, но останется немым.
            mode = "SSRC не сопоставлен, кадры идут как есть"
        elif session is None or not session.ready or version == 0:
            mode = "канал без E2EE, кадры и так открытые"
        elif _passthrough(session, user_id):
            mode = "passthrough, отправитель шлёт открытые кадры"
        else:
            mode = "расшифровка DAVE"

        if modes.get(ssrc) != mode:
            modes[ssrc] = mode
            log.info("ssrc=%s user=%s: %s (DAVE v%s)", ssrc, user_id, mode, version)

    def dave_decrypt(self, packet) -> None:  # noqa: ANN001 — rtp.AudioPacket
        nonlocal failures

        payload = getattr(packet, "decrypted_data", None)
        if not payload:
            # FakePacket, которым буфер затыкает дыру. Расшифровывать нечего,
            # а присваивать ему поле нельзя: у него __slots__ без него.
            return

        state = getattr(self.sink.voice_client, "_connection", None)
        session = getattr(state, "dave_session", None)
        version = getattr(state, "dave_protocol_version", 0)
        user_id = self._cached_id

        report_mode(self.ssrc, session, version, user_id)

        try:
            unwrapped = dave_unwrap(session, version, user_id, davey.MediaType.audio, payload)
        except Exception as exc:
            # Кадр не трогаем: он мог оказаться открытым, и тогда ещё
            # пригодится. Негодный отбракует Opus — ценой одного кадра.
            failures += 1
            if failures == 1 or failures % _DROP_LOG_EVERY == 0:
                log.warning("DAVE не расшифровал кадр (%d-й, ssrc=%s): %s", failures, self.ssrc, exc)
            return

        # Присваиваем, только если кадр правда изменился: у SilencePacket и
        # прочих «ненастоящих» пакетов decrypted_data — атрибут класса.
        if unwrapped is not payload:
            packet.decrypted_data = unwrapped

    def _process_packet(self, packet):  # noqa: ANN001 — сигнатура задана библиотекой
        """То же, что в оригинале, но автор кадра ищется до декодирования.

        У DAVE ключ свой на каждого отправителя, так что без ``_cached_id``
        расшифровать нечем. В оригинале SSRC сопоставлялся с человеком уже
        после ``_decode_packet()`` — то есть всегда поздно.
        """
        if self._cached_id is None:
            # Не перетираем уже известный id: _get_id_from_ssrc() вернёт None,
            # пока не пришло событие speaking, и это стоило бы нам ключа.
            self._cached_id = self.sink.voice_client._get_id_from_ssrc(self.ssrc)
        member = self._get_cached_member()

        dave_decrypt(self, packet)

        pcm = None
        if not self.sink.wants_opus():
            packet, pcm = self._decode_packet(packet)

        data = voice_data(packet, member, pcm=pcm)
        self._last_seq = packet.sequence
        self._last_ts = packet.timestamp
        return data

    decoder._process_packet = _process_packet


def _patch_gap_recovery() -> None:
    """Пункт 2: потерянный пакет не должен уносить с собой весь буфер.

    ``_flag_ready_state`` будит роутер по ``peek()`` — то есть как только в
    джиттер-буфере накопилось больше ``prefsize`` пакетов. А ``pop()`` отдаёт
    пакет только когда взведён ``_has_item``, то есть когда последовательность
    идёт без дыр. Условия разные, и на каждой дыре ``pop()`` возвращает
    ``None``, после чего оригинальный ``_get_next_packet`` делает ``flush()``
    и оставляет один пакет из десяти — «N packets were lost being flushed» в
    логе. На туннеле, где потери обычное дело, до VAD доходят обрывки по 20 мс,
    и речи в них нет.

    Вместо этого перескакиваем дыру: берём следующий по порядку пакет,
    остальные остаются в буфере.
    """
    from discord.ext.voice_recv import opus as recv_opus

    def _get_next_packet(self, timeout: float):
        packet = self._buffer.pop(timeout=timeout)
        if packet is None:
            packet = take_lowest(self._buffer)
            if packet is None:
                return None
        if not packet:
            packet = self._make_fakepacket()
        return packet

    recv_opus.PacketDecoder._get_next_packet = _get_next_packet


def _patch_decoder_resilience() -> None:
    """Пункт 3: сбой на одном кадре стоит кадра, а не всей сессии приёма.

    Исключение из ``pop_data()`` выходит в ``PacketRouter._do_run``, поток
    роутера умирает и в ``finally`` зовёт ``stop_listening()`` — бот остаётся
    в канале навсегда глухим, и снаружи это выглядит как «замолчал сам».

    Ловим здесь всё, а не только ``OpusError``: цена ошибки — один кадр
    (20 мс), цена мёртвого потока — вся сессия. Чтобы заплатка не превратилась
    в глушилку, первое исключение каждого рода пишется целиком, дальше —
    каждое 500-е.
    """
    from discord.ext.voice_recv import opus as recv_opus
    from discord.opus import OpusError

    original = recv_opus.PacketDecoder.pop_data
    dropped = 0

    def pop_data(self, *, timeout: float = 0):
        nonlocal dropped
        try:
            return original(self, timeout=timeout)
        except OpusError as exc:
            dropped += 1
            if dropped == 1 or dropped % _DROP_LOG_EVERY == 0:
                log.warning("пропущен неразобранный голосовой кадр (%d-й): %s", dropped, exc)
        except Exception:
            dropped += 1
            if dropped == 1 or dropped % _DROP_LOG_EVERY == 0:
                log.exception("сбой обработки голосового кадра (%d-й), ssrc=%s", dropped, self.ssrc)
        # None роутер просто пропустит, не трогая синк.
        return None

    recv_opus.PacketDecoder.pop_data = pop_data
