"""Заплатки на приёмный тракт py-cord 2.8.1.

Приём голоса в 2.8.1 не работает ни в каком виде — это признаёт сама
библиотека (``RuntimeWarning`` при ``start_recording`` и issue
`#3139 <https://github.com/Pycord-Development/pycord/issues/3139>`_).
Здесь собрано всё, что нужно поправить, чтобы из канала пошёл чистый PCM.
Патчи ставятся на классы **до** первого подключения к голосу: декриптор
разбирает методы по имени режима в ``__init__``, и после создания объекта
подменять уже поздно.

Что именно чинится
------------------

1. **DAVE выключается.** ``py-cord[voice]`` тянет ``davey``, поэтому в
   IDENTIFY уходит ``max_dave_protocol_version: 1`` и Discord включает
   сквозное шифрование, принимать которое библиотека не умеет. Версию для
   звонка Discord берёт как минимум по участникам, так что объявленный ноль
   просто снимает E2EE, пока бот в канале, — как любой клиент без поддержки
   DAVE. Для нашей задачи это ничего не меняет: бот и так легальный
   участник, который всё расшифровывает и отправляет в STT.

2. **``decrypt_rtp()`` начинает возвращать расшифрованное.** В оригинале
   результат транспортной расшифровки кладётся в локальную переменную и
   теряется, а наружу отдаётся ``packet.decrypted_data`` — оно
   присваивается только внутри ветки DAVE. Без DAVE каждый пакет уходит
   в ``None`` и отбрасывается в ``AudioReader.callback``
   (``if not rtp_packet.decrypted_data: return``), то есть канал молчит.

3. **RTP-расширение срезается по фактической длине.** Оригинал считает
   offset через ``update_extended_header()``, выбрасывает его и режет
   захардкоженные 8 байт. Это верно только когда расширение занимает ровно
   два слова; иначе в Opus уезжает мусор — тот самый
   ``OpusError('corrupted stream')``. Правильная длина — ``length * 4`` из
   заголовка расширения, её и возвращает ``update_extended_header()``.

4. **Ошибка декодирования перестаёт убивать приём.** Исключение из
   ``PacketDecoder.pop_data()`` выходит в ``PacketRouter._do_run``, поток
   роутера умирает и в ``finally`` зовёт ``stop_recording()`` — бот
   остаётся в канале навсегда глухим. Битый пакет должен стоить одного
   пакета, а не сессии.

Пункты 2–4 совпадают по смыслу с форком vito1317/pycord@5a95f98 — диагноз
про двойное срезание расширения найден там. Сам форк не используется: он
ответвлён от master до 2.8.0, а подменять библиотеку целиком в процессе,
который держит токен, — лишний риск.
"""

from __future__ import annotations

import logging

log = logging.getLogger(__name__)

#: Сколько ошибок декодирования пропустить между сообщениями в лог.
_DECODE_ERROR_LOG_EVERY = 500


def apply() -> None:
    """Поставить все заплатки. Зовётся один раз на старте процесса."""
    _disable_dave()
    _patch_rtp_decryption()
    _patch_decoder_resilience()


def _disable_dave() -> None:
    """Объявить Discord'у нулевую версию DAVE — см. пункт 1 в докстринге.

    ``VoiceConnectionState.max_dave_proto_version`` — property поверх
    модульной константы, поэтому правится константа.
    """
    from discord.voice import state as voice_state
    from discord.voice.utils import dependencies as voice_deps

    modules = (voice_deps, voice_state)
    if not any(hasattr(module, "DAVE_PROTOCOL_VERSION") for module in modules):
        log.warning("не нашёл DAVE_PROTOCOL_VERSION — приём голоса может не работать")
        return

    for module in modules:
        module.DAVE_PROTOCOL_VERSION = 0
    log.debug("DAVE отключён: объявлена версия 0")


def _patch_rtp_decryption() -> None:
    """Пункты 2 и 3: отдавать расшифрованное и резать расширение по длине."""
    from discord.voice.receive import reader as voice_reader

    decryptor = getattr(voice_reader, "PacketDecryptor", None)
    if decryptor is None:
        log.warning("не нашёл PacketDecryptor — приём голоса может не работать")
        return

    crypto_error = getattr(voice_reader, "CryptoError", Exception)

    def decrypt_rtp(self, packet):  # noqa: ANN001 — сигнатура задана py-cord
        """Транспортная расшифровка. Ветки DAVE нет: он выключен на входе.

        Если DAVE всё же поднялся (значит `_disable_dave` промахнулся мимо
        новой версии py-cord), в Opus поедет шифротекст — предупреждаем, не
        дожидаясь потока `corrupted stream`.
        """
        if getattr(self.client._connection, "dave_session", None) is not None:
            log.error(
                "DAVE-сессия поднялась вопреки настройке — приём голоса не заработает, "
                "смотрите bashmak/audio/pycord_patch.py"
            )
        packet.decrypted_data = self._decryptor_rtp(packet)
        return packet.decrypted_data

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


def _patch_decoder_resilience() -> None:
    """Пункт 4: битый пакет стоит пакета, а не всей сессии приёма."""
    from discord import opus as discord_opus

    decoder = getattr(discord_opus, "PacketDecoder", None)
    if decoder is None:
        log.warning("не нашёл PacketDecoder — приём голоса не переживёт битый пакет")
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
            if dropped == 1 or dropped % _DECODE_ERROR_LOG_EVERY == 0:
                log.warning("пропущен неразобранный голосовой пакет (%d-й): %s", dropped, exc)
            # None роутер просто пропустит, не трогая синк.
            return None

    decoder.pop_data = pop_data
