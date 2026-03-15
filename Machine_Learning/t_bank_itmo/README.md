# 🏨 Предсказание полноты сведений отельных номеров

> **ML-хакатон Т-Путешествия · Команда OpenTrip**  
> Задача бинарной классификации: определить, достаточно ли данных в описании номера для автоматического матчинга, чтобы не тратить деньги на краудсорсинг.

## 🏆 Результаты

| Метрика | Значение |
|---|---|
| **PR-AUC (Kaggle leaderboard)** | **0.9735** |
| **PR-AUC (валидация, лучшая модель)** | **0.9352** |

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

**Метрика:** `PR-AUC` — полнота при высокой точности детекции критична для бизнеса.

---

## 🗂️ Структура проекта

```
t_bank_itmo/
├── solution.ipynb              # Основной ноутбук с решением
├── public_dataset.csv          # Обучающий датасет (184 138 строк)
├── submission_sample.csv       # Пример формата submission
├── presentation.html           # Презентация решения (15 слайдов)
├── ии чемп опен трип.pdf       # PDF-версия презентации
├── task.md                     # Описание задачи от организаторов
└── README.md
```

---

## 🧠 Архитектура решения

### Пайплайн

```
public_dataset.csv (184 138 строк)
        ↓
1. Предобработка (fillna, длина описания, кол-во слов)
        ↓
2. Feature Engineering: AMENITY_GROUPS → бинарные флаги + completeness_score
        ↓
3. Словарный скор покрытия (vocab_coverage)
        ↓
4. Hotel-статистики из train (avg_name_len, avg_words, target_rate, room_count)
        ↓
5. Устранение мультиколлинеарности (порог корреляции > 0.9)
        ↓
6. TF-IDF (5000 признаков, ngram 1-2) → TruncatedSVD (50 компонент)
        ↓
7. Train / Val split 80/20 (stratify, random_state=42)
        ↓
8. Обучение 3 моделей: Logistic Regression, Random Forest, LightGBM
        ↓
9. Лучшая модель → submission.csv
```

---

## 🔧 Feature Engineering

### AMENITY_GROUPS
Словарь из ~20 групп признаков с ключевыми словами на русском и английском:

| Группа | Примеры ключевых слов |
|---|---|
| `standard` | стандарт, standard, эконом, classic |
| `comfort` | комфорт, superior, улучшенный |
| `deluxe` | делюкс, deluxe, de luxe |
| `suite` | люкс, suite, апартамент, студия |
| `single_bed` | одноместный, single, sgl |
| `double_bed` | двухместный, double, dbl |
| `twin_beds` | twin, две кровати, раздельные |
| `king_bed` | king, king size |
| `sea_view` | вид на море, sea view, seaview |
| `city_view` | вид на город, city view |
| ... | ... |

### Критические группы (has_* флаги)
- `has_room_level` — уровень номера (standard/deluxe/suite...)
- `has_bed_config` — конфигурация кровати
- `has_view` — вид из окна
- `has_bedroom_count` — количество комнат
- `has_outdoor` — балкон/терраса
- `has_meal` — питание

`completeness_score` = сумма всех `has_*` (от 0 до 6)

### Топ признаков по корреляции с таргетом

| Признак | Корреляция |
|---|---|
| `hotel_target_rate` | 0.51 |
| `room_name_len` | 0.24 |
| `hotel_avg_name_len` | 0.23 |
| `double_bed` | 0.17 |
| `single_bed` | 0.17 |
| `sofa_bed` | 0.17 |
| `completeness_score` | 0.09 |

---

## ⚙️ Модели и гиперпараметры

### Сравнение моделей

| Модель | PR-AUC (val) |
|---|---|
| Logistic Regression | 0.8815 |
| Random Forest | 0.9109 |
| **LightGBM** | **0.9352** ✅ |

### Лучшие параметры LightGBM

| Параметр | Значение |
|---|---|
| `n_estimators` | 300 |
| `learning_rate` | 0.1 |
| `max_depth` | 5 |
| `num_leaves` | 20 |
| `min_child_samples` | 15 |

### Поиск гиперпараметров
`RandomizedSearchCV`, `n_iter=30`, `cv=3`, метрика `PR-AUC`

---

## 📊 Данные

| Параметр | Значение |
|---|---|
| Обучающая выборка | 184 138 строк |
| Train / Val split | 80% / 20% |
| Признаков итого | 82 |
| Дисбаланс классов | ~0.94 (label=1 чуть больше) |

---

## 🔍 Интерпретируемость (Feature Importance)

Топ признаков по важности (LightGBM):

1. `hotel_target_rate` — доля несматченных комнат в отеле
2. `hotel_avg_vocab_coverage` — средний словарный охват по отелю
3. `hotel_room_count` — количество комнат в отеле
4. `hotel_avg_name_len` — средняя длина описания по отелю
5. `room_name_len` — длина описания конкретной комнаты
6. `tfidf_svd_0..N` — TF-IDF компоненты

**Вывод:** модель сильно опирается на **внутриотельную специфику** (`hotel_target_rate`, `hotel_avg_*`) — отели с плохим контентом дают плохие описания системно.

---

## 🚀 Запуск

### Зависимости

```bash
pip install pandas numpy scikit-learn lightgbm shap matplotlib seaborn
```

### Запуск ноутбука

```bash
jupyter notebook solution.ipynb
```

Данные должны лежать рядом:
```
t_bank_itmo/
├── public_dataset.csv
└── submission_sample.csv
```


---

## 🛠️ Стек технологий

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![LightGBM](https://img.shields.io/badge/LightGBM-2CA5E0?style=for-the-badge)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![SHAP](https://img.shields.io/badge/SHAP-FF6F00?style=for-the-badge)
