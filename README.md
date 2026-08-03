# Башмак

Голосовой бот для Discord. Слушает нескольких участников одновременно,
отзывается на своё имя, разговаривает и включает музыку. Всё локально, всё на
CPU: llama.cpp + faster-whisper + Piper + Silero VAD.

Техническое задание — [bashmak_project_plan.md](bashmak_project_plan.md).

---

## Развёртывание на чистом сервере

Целевое железо: Ubuntu x86_64, 6 ядер, 40 ГБ RAM. GPU не нужен.

```bash
git clone <репозиторий> bashmak
cd bashmak
bash scripts/setup.sh --systemd
```

> Запуск именно через `bash scripts/...` — не опечатка: если репозиторий
> пушили с Windows, бит исполняемости в git-индексе не сохранился. Первым же
> шагом `setup.sh` проставит его сам. Чтобы это не повторялось, один раз
> зафиксируйте бит в индексе: `git update-index --chmod=+x scripts/*.sh`.

Скрипт делает всё сам: системные пакеты (`ffmpeg`, `libopus`, сборочные
инструменты), venv с зависимостями, готовые бинарники llama.cpp и загрузку
всех весов — LLM, STT, TTS, VAD. Занимает 10–30 минут и около 10 ГБ диска,
в основном на модель.

Дальше — токен бота:

```bash
nano .env      # DISCORD_TOKEN=...
```

Токен берётся на [портале разработчика Discord](https://discord.com/developers/applications):
вкладка **Bot** → *Reset Token*. Там же включите privileged intent
**SERVER MEMBERS** (нужен, чтобы бот обращался к людям по нику) и в **OAuth2 →
URL Generator** выдайте боту scope `bot` + `applications.commands` и права
*Connect*, *Speak*, *Use Voice Activity*.

Проверка, что всё поднялось:

```bash
./scripts/doctor.sh
```

Скрипт печатает таблицу PASS/FAIL: пакеты, файлы моделей, ответ
`llama-server`, сквозной круг Piper→Whisper и валидность токена. Если всё
зелёное — запускаем:

```bash
sudo systemctl start bashmak
journalctl -u bashmak -f
```

`bashmak.service` зависит от `llama-server.service`, оба включены в автозапуск
и перезапускаются при падении.

### Флаги setup.sh

| Флаг | Зачем |
|---|---|
| `--profile fast\|balanced\|quality` | Какие модели качать (по умолчанию `balanced` — Qwen2.5-7B + whisper small) |
| `--skip-models` | Только окружение, без весов |
| `--models-only` | Только докачать веса в готовое окружение |
| `--systemd` | Установить и включить юниты |
| `--force` | Пересобрать venv/llama.cpp и перекачать модели |

Повторный запуск безопасен: всё уже сделанное помечается `[skip]`.

### Профили

| Профиль | LLM | STT | Когда |
|---|---|---|---|
| `fast` | Qwen2.5-3B Q4_K_M | whisper small | Важна скорость, участников много |
| `balanced` | Qwen2.5-7B Q4_K_M | whisper small | По умолчанию |
| `quality` | Qwen2.5-7B Q4_K_M | whisper medium | Качество важнее задержки |

Профиль влияет и на то, что скачается, и на пути в `config.yaml`. Сменить
после установки: `./scripts/setup.sh --profile quality --models-only`, затем
поправить пути в `config.yaml`.

---

## Если Discord заблокирован по IP

Проверить, есть ли проблема:

```bash
curl -sS -o /dev/null -w 'connect=%{time_connect} tls=%{time_appconnect} code=%{http_code}\n' --max-time 10 https://discord.com/api/v10/gateway
```

`connect=0` означает, что SYN дропается и до TLS дело не доходит. Обход на
уровне DPI (zapret и подобные) тут бесполезен: он ломает разбор TLS
ClientHello внутри уже установленного соединения, а соединения нет. Нужен
туннель.

Заворачивать весь сервер не надо — `llama-server` слушает `127.0.0.1`,
а STT/TTS/VAD сети не касаются вообще. Наружу должен ходить только бот:
веб-сокет шлюза и голосовой UDP, суммарно меньше сотни килобит.

**Почему именно TUN, а не SOCKS.** discord.py умеет проксировать REST и шлюз,
но голос открывает UDP-сокет напрямую, мимо настроек прокси. С прокси бот
зайдёт в канал и будет молчать в обе стороны. Нужен виртуальный интерфейс,
через который уходит весь трафик процесса, включая UDP.

Порядок:

1. На VPS в панели (3x-ui и аналоги) завести inbound **VLESS + TCP +
   REALITY**, flow `xtls-rprx-vision`. Именно TCP+Vision: XHTTP и прочие
   маскирующиеся транспорты добавляют джиттер, а голос чувствителен к
   разбросу задержки сильнее, чем к её величине.
2. На сервере поставить sing-box. Через `apt` с `deb.sagernet.org` в
   заблокированной сети обычно не выходит — репозиторий на AWS и отваливается
   по таймауту так же, как Discord. Ставим бинарником с GitHub:

```bash
sudo ./scripts/install_singbox.sh
```

3. Вписать свои значения (адрес VPS, uuid, public_key), проверить и поднять:

```bash
sudo nano /etc/sing-box/config.json && sudo sing-box check -c /etc/sing-box/config.json
```

```bash
sudo systemctl enable --now sing-box && ip -brief addr show tun-bashmak
```

   Проверить канал до VPS отдельно от маршрутизации — в конфиге для этого
   есть SOCKS-инбаунд. Здесь должно быть `code=200`:

```bash
curl -sS --socks5-hostname 127.0.0.1:10808 --max-time 15 -o /dev/null -w 'code=%{http_code}\n' https://discord.com/api/v10/gateway
```

4. Перегенерировать юнит с включённой маркировкой трафика:

```bash
./scripts/setup.sh --tunnel
```

   Юниты собираются из `deploy/*.template` в момент установки, так что после
   любой правки шаблона (или обновления репозитория) `--systemd`/`--tunnel`
   надо прогнать заново — иначе в `/etc/systemd/system` останется копия
   времён прошлой установки.

[deploy/tunnel.sh](deploy/tunnel.sh) маркирует трафик по cgroup юнита —
не по пользователю. Поэтому не нужно заводить отдельного пользователя и
переразбивать права на каталог проекта: правило бьёт ровно в один сервис,
а ssh и apt остаются на прямом маршруте. Локальные адреса и DNS из тоннеля
исключены — иначе бот потерял бы собственный `llama-server` на `127.0.0.1`.

Проверить, что получилось:

```bash
sudo ./deploy/tunnel.sh status
```

> **Если после поднятия туннеля на машине умер DNS** (`git`, `apt`, всё
> подряд пишут «Could not resolve host») — sing-box зарегистрировал себя в
> systemd-resolved как резолвер для линка `tun-bashmak` с доменом `~.`,
> то есть перехватил запросы всего хоста, а не только бота.
> `tunnel.sh up` это снимает, но перехват возвращается при каждом
> перезапуске самого sing-box. Разово лечится так:
>
> ```bash
> sudo resolvectl revert tun-bashmak
> ```
>
> Увидеть, вернулся ли перехват, можно в `tunnel.sh status` — строка про
> DNS на туннеле должна быть пустой.

После этого `./scripts/doctor.sh` должен показать `DISCORD_TOKEN` зелёным —
именно эта проверка первой упирается в блокировку.

---

## Как этим пользоваться

В Discord: зайдите в голосовой канал и вызовите `/start`. Дальше просто
говорите, обращаясь по имени:

- «**Башмак**, расскажи анекдот» — ответит голосом;
- «**Башмак**, включи Кино Группа крови» — найдёт и включит;
- «**Башмак**, следующий» / «поставь на паузу» / «сделай потише» / «выключи музыку»;
- реплики без имени бот не обрабатывает, но запоминает как контекст разговора.

Команды: `/start`, `/leave`, `/status`, `/forget` (сбросить историю диалога).

Пока бот отвечает, музыка автоматически приглушается и возвращается к прежней
громкости — трек не останавливается и не теряет позицию.

---

## Устройство

```
Discord voice (по SSRC, раздельно на каждого)
   │
   ├─ sink.py ─ буфер ─ VAD ─┐
   ├─ sink.py ─ буфер ─ VAD ─┼─► пул процессов STT (параллельно)
   └─ sink.py ─ буфер ─ VAD ─┘            │
                                          ▼
                              wake word «башмак» (fuzzy)
                                          │
                                   intent-роутер
                              ┌───────────┴───────────┐
                         музыка                     диалог
                         (быстрый путь)      очередь к LLM (по одному)
                              │                       │
                              │                     Piper TTS
                              └────────┬──────────────┘
                                 Output Arbiter
                             (микс: речь + приглушённая музыка)
                                       │
                              один voice-send в канал
```

Асимметрия здесь принципиальная: **приём** параллелится по говорящим,
**инференс LLM** — нет (на CPU это только замедлит), **выход** — тоже один.
Поэтому реплики не теряются, а выстраиваются в очередь по времени окончания
фразы.

| Модуль | Что делает |
|---|---|
| [audio/sink.py](bashmak/audio/sink.py) | Приём PCM из voice-recv, разводка по говорящим |
| [audio/voice_recv_patch.py](bashmak/audio/voice_recv_patch.py) | Расшифровка DAVE и живучесть приёма поверх библиотеки |
| [audio/buffer.py](bashmak/audio/buffer.py) | Потокобезопасные буферы, граница «поток приёма ↔ asyncio» |
| [audio/vad.py](bashmak/audio/vad.py) | Silero VAD в onnxruntime, нарезка на фразы |
| [audio/listener.py](bashmak/audio/listener.py) | Опрос буферов, склейка VAD → STT |
| [stt/whisper_worker.py](bashmak/stt/whisper_worker.py) | faster-whisper в пуле процессов |
| [wakeword/filter.py](bashmak/wakeword/filter.py) | Нечёткий поиск имени в тексте |
| [intent/router.py](bashmak/intent/router.py) | Регекс + LLM-фоллбэк: болтовня или команда |
| [llm/queue_manager.py](bashmak/llm/queue_manager.py) | Очередь к llama-server, история диалога |
| [tts/piper_worker.py](bashmak/tts/piper_worker.py) | Piper в пуле процессов, синтез по предложениям |
| [music/player.py](bashmak/music/player.py) | Очередь треков поверх микшера |
| [output/arbiter.py](bashmak/output/arbiter.py) | Единственный голосовой выход, даккинг музыки |

---

## Настройка

Всё в [config.yaml](config.example.yaml) — пороги VAD, размеры моделей, число
потоков, промпт-персона в [llm/persona.py](bashmak/llm/persona.py).

Бюджет ядер по умолчанию рассчитан на 6 физических: `llm.threads: 4`,
`stt.workers: 2` по 2 потока, VAD и остальное — на основном лупе. Если бот
тормозит при трёх и более говорящих, снижайте `llm.threads`, а не
`stt.workers`: распознавание должно успевать, иначе речь начнёт теряться в
буферах.

## Диагностика

Каждой реплике присваивается correlation id, и все стадии пишут свою
длительность — по логу видно, где именно затык:

```
12:31:04 INFO [u4711@812.3] ...whisper_worker: stt: 640 мс (user=4711, sec=2.1, chars=34)
12:31:04 INFO [u4711@812.3] ...bot: обращение от Вася (score=100): 'расскажи анекдот'
12:31:06 INFO [u4711@812.3] ...queue_manager: llm: 2140 мс (queue=0, chars=118)
12:31:07 INFO [u4711@812.3] ...piper_worker: tts[0]: 310 мс (chars=61)
```

Логи: `journalctl -u bashmak -f` и `logs/bashmak.log` (с ротацией).

## Тесты

```bash
.venv/bin/pytest
```

Тесты не требуют ни сети, ни моделей, ни Discord — проверяется чистая логика:
автомат VAD, нечёткий wake word, регексы intent-роутера, окно истории диалога
и микшер (включая приоритет речи, даккинг и поведение при мёртвом голосовом
выходе).
