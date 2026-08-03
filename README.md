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

## Как этим пользоваться

В Discord: зайдите в голосовой канал и вызовите `/join`. Дальше просто
говорите, обращаясь по имени:

- «**Башмак**, расскажи анекдот» — ответит голосом;
- «**Башмак**, включи Кино Группа крови» — найдёт и включит;
- «**Башмак**, следующий» / «поставь на паузу» / «сделай потише» / «выключи музыку»;
- реплики без имени бот не обрабатывает, но запоминает как контекст разговора.

Команды: `/join`, `/leave`, `/status`, `/forget` (сбросить историю диалога).

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
| [audio/sink.py](bashmak/audio/sink.py) | Приём PCM из py-cord, разводка по говорящим |
| [audio/buffer.py](bashmak/audio/buffer.py) | Потокобезопасные буферы, граница «поток py-cord ↔ asyncio» |
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
.venv/bin/python -m pytest
```

Тесты не требуют ни сети, ни моделей, ни Discord — проверяется чистая логика:
автомат VAD, нечёткий wake word, регексы intent-роутера и микшер.
