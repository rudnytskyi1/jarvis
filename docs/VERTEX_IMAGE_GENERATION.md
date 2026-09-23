# Генерация картинок через Vertex AI (кредиты Google Cloud)

Владелец 2026-09-23: «для генерации картинок теперь используй vertexai api (у
меня бесплатные 300$ credits)». Этот документ — про вторую дорогу к той же
модели Nano Banana. Первая (`provider: gemini`) осталась на месте: она
работает по одной долгоживущей `GEMINI_API_KEY` и у владельца отвечает
`HTTP 402 Payment Required`, потому что баланс ключа пуст.

Что различается между дорогами:

| | `provider: gemini` | `provider: vertex` |
|---|---|---|
| Ключ | `GEMINI_API_KEY` (AI Studio) | сервис-аккаунт Google Cloud или Express-ключ |
| Адрес | `generativelanguage.googleapis.com` | `{region}-aiplatform.googleapis.com` |
| Кто платит | аккаунт ключа | проект Google Cloud (кредиты) |
| Срок жизни доступа | ключ бессрочный | OAuth2-токен на 1 час, обновляется сам |
| Тело запроса, учёт расхода, повторы | одинаковые | одинаковые |

## Что нужно с одной стороны

1. Проект Google Cloud, у которого есть кредиты (у владельца — бесплатные
   $300) и включённый биллинг.
2. В проекте включён **Vertex AI API**:
   *Console → APIs & Services → Enable APIs → Vertex AI API*. Без этого
   Google отвечает `403 SERVICE_DISABLED`.
3. Сервис-аккаунт с ролью **Vertex AI User** и скачанным JSON-ключом:
   *Console → IAM & Admin → Service Accounts → Create → Keys → Add key →
   Create new key → JSON*. Файл скачивается один раз: второй раз Google его не
   покажет.

   Ключ — это секрет. Он лежит в `data/` (каталог в `.gitignore`) или в любом
   месте вне репозитория; в YAML и в чат он не попадает.
4. Идентификатор проекта (`project id`, например `rowan-images-482301`) —
   он отличается от отображаемого имени.

## Настройка в Rowan

1. Положить ключ туда, где хаб его ждёт:

   ```powershell
   pwsh -File scripts\set-vertex-key.ps1 -KeyPath "$env:USERPROFILE\Downloads\rowan-images-482301-1a2b3c.json"
   ```

   Скрипт копирует файл в `data/vertex-credentials.json`, проверяет тип
   (`service_account`), печатает адрес сервис-аккаунта и **не** печатает
   закрытый ключ. Если файл уже лежит в другом месте — путь можно не копировать
   и указать его в конфиге (`vertex_credentials_path`).

2. В активном конфиге (`config.openai.yaml`) указать проект и регион:

   ```yaml
   server:
     image_generation:
       enabled: true
       provider: vertex            # было: gemini
       model: gemini-3.1-flash-image
       vertex_project: rowan-images-482301   # project id, не имя
       vertex_location: global               # или us-central1
   ```

   Регион `global` — общий узел Vertex, там новые модели появляются первыми;
   `us-central1` — обычный региональный узел. Если модель в выбранном регионе
   недоступна, Google ответит `404 Publisher Model ... not found`, и его
   сообщение показывается в Telegram и в панели как есть.

3. Перезапустить хаб (`start-jarvis-openai.bat`) — комната переподключится
   сама, на клиентах ничего менять не нужно.

## Как проверить, что всё работает

* `python scripts/vertex_image_probe.py` — один платный запрос без камеры и без
  телефона: скрипт печатает, откуда он взял доступ, куда пошёл и что вернулось.
* `python scripts/check_image_generation.py` — то же самое в общем чекере,
  которым проверяли Nano Banana (`--reference photo.jpg` добавит правку
  готового фото).
* `http://127.0.0.1:8770/health` — поле `image_generation: true` означает, что
  провайдер включён **и** доступ у него есть (это про конфигурацию, не про
  платный запрос).

## Как это устроено в коде

| Файл | Что делает |
|---|---|
| `hub/vertex_auth.py` | доступ: Express-ключ, готовый токен, сервис-аккаунт (подписанный JWT), файл `gcloud auth application-default login`; токен кэшируется на час |
| `hub/image_generation.py` | выбор дороги (`_endpoint`), тело запроса, учёт расхода, повторы, разбор ответа и отказа |
| `common/config.py` | `ImageGenerationConfig`: `provider` и поля `vertex_*` |
| `scripts/set-vertex-key.ps1` | положить ключ на место и проверить его |
| `scripts/vertex_image_probe.py` | одна живая проверка с понятным отчётом |

Порядок поиска доступа: `VERTEX_API_KEY` (Express-режим, ключ прямо в запросе)
→ `VERTEX_ACCESS_TOKEN` (готовый токен) → `GOOGLE_APPLICATION_CREDENTIALS` →
`data/vertex-credentials.json` → файл `gcloud auth application-default login`.
Первый найденный и используется.

Ошибки называются своими словами и без догадок:

* `401/403` — «Vertex AI отклонил доступ или модель»: смотреть роль
  сервис-аккаунта и включён ли Vertex AI API в проекте;
* `402` — «проект не может заплатить»: кредиты/биллинг, переформулировка
  запроса не поможет;
* `404` — Google печатает своё `Publisher Model ... not found`: не тот регион
  или не тот id модели;
* нет проекта или нет файла ключа — запрос вообще не уходит, в ответе сказано,
  чего именно не хватает.
