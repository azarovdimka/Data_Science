# 🏨 Предсказание полноты сведений отельных номеров

> **ML-хакатон Т-Путешествия · Команда OpenTrip**  
> Задача бинарной классификации: определить, достаточно ли данных в описании номера для автоматического матчинга, чтобы не тратить деньги на краудсорсинг.

## 🏆 Результаты

| Метрика | Значение |
|---|---|
| **PR-AUC (Kaggle leaderboard)** | **0.9735** |
| **PR-AUC (скрытый тест)** | **0.9438** |

---

## 📋 Бизнес-контекст

Сервис **Т-Путешествия** ежедневно обрабатывает тысячи предложений от поставщиков. Каждое предложение нужно сопоставить с **мастер-комнатой** в единой базе. Если автоматический матчинг не срабатывает — комната уходит в **краудсорсинг**, где люди вручную ищут совпадение. Это дорого.

**Проблема:** часто поставщики дают неполные описания — например, просто «люкс» без указания вида из окна. Люди в краудсорсинге тоже не могут решить задачу, но деньги уже потрачены.

**Решение:** заранее детектировать такие «пустые» комнаты и не отправлять их в краудсорсинг.

### Воркфлоу матчинга

```
Поставщик → определение hotel_id → ML-матчинг с мастер-комнатой
                                          ↓
                              Уверенный матч → в БД ✅
                                          ↓
                              Нет матча → краудсорсинг 💸
                                          ↓
                         [НАША ЗАДАЧА] Нет данных → задержать ⛔
```

---

## 🎯 Постановка задачи

**Бинарная классификация:**
- `label = 1` — мало данных, комната **не сматчена** → задержать, не отправлять в краудсорсинг
- `label = 0` — данных достаточно, комната **сматчена** → в базу на разметку

**Входные данные:**
- `supplier_room_name` — название номера от поставщика
- `hotel_id` — идентификатор отеля

**Метрика:** `PR-AUC` — полнота при высокой точности детекции критична для бизнеса (недовыдача понятных комнат в краудсорсинг = прямые потери выручки).

---

## 🗂️ Структура проекта

```
t_bank_itmo/
├── solution.ipynb              # Основной ноутбук с решением
├── public_dataset.csv          # Обучающий датасет
├── submission_sample.csv       # Пример формата submission
├── presentation.html           # Презентация решения (15 слайдов)
├── ии чемп опен трип.pdf       # PDF-версия презентации
├── task.md                     # Описание задачи от организаторов
└── README.md
```

---

## 🧠 Архитектура решения

### Выбор подхода

| Подход | Решение |
|---|---|
| ❌ TF-IDF + классик | Не понимает семантику. «twin» и «двуспальный» — разные токены |
| ✅ **RuBERT + Hotel Embeddings** | Понимает контекст и морфологию русского языка |
| ⚠️ LLM (GPT/LLaMA) | Избыточно, медленно, дорого. Задача не требует генерации |

### Модель RoomClassifier

```
supplier_room_name → Токенизатор RuBERT (MAX_LEN=128)
                              ↓
                     RuBERT Encoder
                              ↓
                    CLS-вектор [768 dim]
                              ↓                    hotel_id → Embedding [16 dim]
                         Concat [784 dim] ←────────────────────────────────────
                              ↓
                    Dropout(0.3) → Linear(784→1)
                              ↓
                    Вероятность label=1
```

**Ключевые компоненты:**
- **`DeepPavlov/rubert-base-cased`** — русскоязычная BERT-модель, понимает морфологию (падежи, суффиксы), семантику («двуспальная» ≈ «double»)
- **Hotel Embedding** — каждый отель получает обучаемый 16-мерный вектор. «Standard» в 5★ отеле ≠ «Standard» в хостеле
- **`view(-1)` вместо `squeeze()`** — безопасность при batch_size=1

---

## ⚙️ Гиперпараметры

| Параметр | Значение |
|---|---|
| Модель | `DeepPavlov/rubert-base-cased` |
| MAX_LEN | 128 |
| BATCH_SIZE | 32 |
| EPOCHS | 15 |
| LR | 2e-5 |
| Optimizer | AdamW |
| Scheduler | Linear warmup (10%) |
| Loss | BCEWithLogitsLoss |
| Dropout | 0.3 |
| Hotel emb dim | 16 |

---

## 🔧 Технические решения

### Mixed Precision Training
```python
with torch.amp.autocast('cuda'):
    outputs = model(ids, mask, h_idx).view(-1)
    loss = criterion(outputs, targets)

scaler.scale(loss).backward()
scaler.step(optimizer)
```
Ускорение обучения ×2, экономия GPU-памяти до 50% без потери качества.

### Linear Warmup Scheduler
Первые 10% шагов LR плавно растёт от 0 до 2e-5, затем линейно убывает. Защищает предобученные веса BERT от резкого изменения в начале обучения.

### Обработка OOV-отелей
```python
num_hotels_with_unknown = num_train_hotels + 1  # +1 для неизвестных отелей

test_df['hotel_idx'] = test_df['hotel_id'].apply(
    lambda x: hotel_mapping.get(x, num_train_hotels)  # OOV → специальный индекс
)
```

### Сохранение лучшей модели
```python
if pr_auc > best_pr_auc:
    torch.save(model.state_dict(), 'models/best_model.pth')
```
Сохраняется лучшая по PR-AUC на валидации, а не последняя эпоха.

---

## 📊 Пайплайн данных

```
public_dataset.csv
        ↓
LabelEncoder (hotel_id → числовой индекс)
        ↓
Train / Val split 80/20 (random_state=42)
        ↓
RoomDataset (токенизация, padding до MAX_LEN)
        ↓
DataLoader (batch=32, shuffle=True для трейна)
        ↓
RoomClassifier (RuBERT + Hotel Embedding)
        ↓
BCEWithLogitsLoss → AdamW → Linear Warmup
        ↓
Валидация → PR-AUC → сохранение лучшей модели
        ↓
Предсказание на тесте → submission.csv
```

---

## 📈 Результаты обучения

| Эпоха | PR-AUC (val) |
|---|---|
| 1 | 0.9255 |
| 2 | 0.9376 |
| 3 | 0.9399 |
| 6 | **0.9438** ← финальный тест |
| Kaggle | **0.9735** |

---

## 🔍 Интерпретируемость модели

Модель научилась:
- **Игнорировать шумовые слова:** «завтрак включён», «2 комнаты», «питание для детей не включено»
- **Фокусироваться на ключевых маркерах:** «superior», «sea view», «king bed», «deluxe», «twin», «balcony»
- **Учитывать внутриотельную специфику** через Hotel Embedding

---

## 🚀 Запуск

### Зависимости

```bash
pip install torch transformers pandas numpy scikit-learn tqdm joblib
```

### Обучение и генерация submission

```bash
# Структура данных
data/
├── public_dataset.csv
└── new_submission_sample.csv

# Запуск
python solution.py
```

После обучения:
- `models/best_model.pth` — лучшая модель
- `models/hotel_label_encoder.pkl` — энкодер отелей
- `models/training_log.txt` — лог обучения
- `submission.csv` — предсказания для теста

---

## 💡 Возможные улучшения

- **Аугментация данных** — перефразирование названий номеров
- **Focal Loss** — для несбалансированных классов
- **Ensemble** — RuBERT + gradient boosting на ручных признаках
- **Анализ attention weights** — для более глубокой интерпретации
- **Стратифицированный split** по hotel_id

---

## 👥 Команда OpenTrip

| Участник | Telegram |
|---|---|
| Лилиана Валиева | [@seym0](https://t.me/seym0) |
| Константин Гене | [@k_genne](https://t.me/k_genne) |
| Дмитрий Азаров | [@Azarov_ML](https://t.me/Azarov_ML) |

---

## 🛠️ Стек технологий

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![HuggingFace](https://img.shields.io/badge/HuggingFace-FFD21E?style=for-the-badge&logo=huggingface&logoColor=black)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
