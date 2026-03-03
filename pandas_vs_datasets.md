# Использование библиотеки Datasets с Pandas

Это краткое введение в использование библиотеки `datasets` совместно с Pandas, с акцентом на обработку данных с помощью функций Pandas и конвертацию между форматами.

Это особенно полезно, так как позволяет выполнять быстрые операции: `datasets` использует PyArrow под капотом, который хорошо интегрирован с Pandas.

## Формат Dataset

По умолчанию datasets возвращают обычные объекты Python: целые числа, числа с плавающей точкой, строки, списки и т.д.

Чтобы получить вместо этого Pandas DataFrame или Series, можно установить формат датасета в `pandas` с помощью [Dataset.with_format()](/docs/datasets/v4.6.1/en/package_reference/main_classes#datasets.Dataset.with_format):

```py
>>> from datasets import Dataset
>>> data = {"col_0": ["a", "b", "c", "d"], "col_1": [0., 0., 1., 1.]}
>>> ds = Dataset.from_dict(data)
>>> ds = ds.with_format("pandas")
>>> ds[0]       # pd.DataFrame
  col_0  col_1
0     a    0.0
>>> ds[:2]      # pd.DataFrame
  col_0  col_1
0     a    0.0
1     b    0.0
>>> ds["col_0"]  # pd.Series
0    a
1    b
2    c
3    d
Name: col_0, dtype: object
```

Это также работает для объектов `IterableDataset`, полученных, например, с помощью `load_dataset(..., streaming=True)`:

```py
>>> ds = ds.with_format("pandas")
>>> for df in ds.iter(batch_size=2):
...     print(df)
...     break
  col_0  col_1
0     a    0.0
1     b    0.0
```

## Обработка данных

Функции Pandas обычно быстрее, чем обычные функции Python, написанные вручную, поэтому они являются хорошим вариантом для оптимизации обработки данных. Вы можете использовать функции Pandas для обработки датасета в [Dataset.map()](/docs/datasets/v4.6.1/en/package_reference/main_classes#datasets.Dataset.map) или [Dataset.filter()](/docs/datasets/v4.6.1/en/package_reference/main_classes#datasets.Dataset.filter):

```python
>>> from datasets import Dataset
>>> data = {"col_0": ["a", "b", "c", "d"], "col_1": [0., 0., 1., 1.]}
>>> ds = Dataset.from_dict(data)
>>> ds = ds.with_format("pandas")
>>> ds = ds.map(lambda df: df.assign(col_2=df.col_1 + 1), batched=True)
>>> ds[:2]
  col_0  col_1  col_2
0     a    0.0    1.0
1     b    0.0    1.0
>>> ds = ds.filter(lambda df: df.col_0 == "b", batched=True)
>>> ds[0]
  col_0  col_1  col_2
0     b    0.0    1.0
```

Мы используем `batched=True`, потому что обрабатывать данные пакетами в Pandas быстрее, чем построчно. Также можно использовать `batch_size=` в `map()` для установки размера каждого `df`.

Это также работает для [IterableDataset.map()](/docs/datasets/v4.6.1/en/package_reference/main_classes#datasets.IterableDataset.map) и [IterableDataset.filter()](/docs/datasets/v4.6.1/en/package_reference/main_classes#datasets.IterableDataset.filter).

## Импорт или экспорт из Pandas

Для импорта данных из Pandas используйте [Dataset.from_pandas()](/docs/datasets/v4.6.1/en/package_reference/main_classes#datasets.Dataset.from_pandas):

```python
ds = Dataset.from_pandas(df)
```

Для экспорта Dataset в Pandas DataFrame используйте [Dataset.to_pandas()](/docs/datasets/v4.6.1/en/package_reference/main_classes#datasets.Dataset.to_pandas):

```python
df = ds.to_pandas()
```

---

## 🔍 Ключевые различия между DataFrame (Pandas) и Dataset (Datasets/Transformers)

### 📚 **1. Происхождение библиотек**

| Аспект | Pandas DataFrame | Datasets Dataset |
|--------|------------------|------------------|
| **Библиотека** | `pandas` | `datasets` (Hugging Face) |
| **Назначение** | Универсальная обработка табличных данных | Специализирован для ML/NLP задач |
| **Год создания** | 2008 | 2020 |

### ⚙️ **2. Внутренняя архитектура**

**Pandas DataFrame:**
- Использует NumPy массивы в качестве основы
- Хранит данные в оперативной памяти (RAM)
- Ограничен размером доступной памяти

**Datasets Dataset:**
- Использует Apache Arrow (колоночный формат)
- Поддерживает memory mapping (данные на диске, доступ как из памяти)
- Может работать с данными, превышающими объем RAM
- Оптимизирован для больших датасетов

### 🚀 **3. Производительность**

```python
# Pandas - все в памяти
df = pd.read_csv('large_file.csv')  # Загружает ВСЕ в RAM

# Datasets - ленивая загрузка
ds = load_dataset('csv', data_files='large_file.csv')  # Загружает по требованию
```

**Преимущества Dataset:**
- Быстрее при работе с большими файлами (>1GB)
- Эффективное кэширование операций
- Параллельная обработка из коробки

### 🔄 **4. Операции с данными**

**Pandas - императивный стиль:**
```python
df['new_col'] = df['old_col'].apply(lambda x: x * 2)
df = df[df['value'] > 10]
```

**Datasets - функциональный стиль:**
```python
ds = ds.map(lambda x: {'new_col': x['old_col'] * 2})
ds = ds.filter(lambda x: x['value'] > 10)
```

### 💾 **5. Кэширование**

**Datasets автоматически кэширует результаты:**
```python
# Первый запуск - выполняется
ds = ds.map(expensive_function)  

# Второй запуск - берется из кэша!
ds = ds.map(expensive_function)  
```

**Pandas не кэширует:**
```python
# Каждый раз выполняется заново
df = df.apply(expensive_function)
df = df.apply(expensive_function)  # Повторное вычисление
```

### 🎯 **6. Интеграция с ML-экосистемой**

**Dataset:**
- Нативная интеграция с Transformers
- Прямая загрузка в PyTorch/TensorFlow
- Встроенная поддержка токенизации

```python
from datasets import load_dataset
from transformers import AutoTokenizer

ds = load_dataset('imdb')
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
ds = ds.map(lambda x: tokenizer(x['text']), batched=True)
```

**DataFrame:**
- Требует ручной конвертации
- Нужны дополнительные шаги для ML

### 📊 **7. Индексация и доступ к данным**

**Pandas:**
```python
df.loc[0]           # По метке
df.iloc[0]          # По позиции
df['column']        # Колонка
df[['col1', 'col2']] # Несколько колонок
```

**Datasets:**
```python
ds[0]               # Одна строка (dict)
ds[:10]             # Срез (dict of lists)
ds['column']        # Колонка (list)
ds.select([0, 1, 2]) # Выбор строк
```

### ⚠️ **8. На что обратить внимание при переходе**

#### **Типы возвращаемых данных:**

```python
# Pandas
df[0]  # → pd.Series (одна строка)
df[:2] # → pd.DataFrame (несколько строк)

# Datasets (по умолчанию)
ds[0]  # → dict (одна строка)
ds[:2] # → dict of lists (несколько строк)

# Datasets (с форматом pandas)
ds = ds.with_format('pandas')
ds[0]  # → pd.DataFrame (одна строка!)
ds[:2] # → pd.DataFrame (несколько строк)
```

#### **Изменяемость:**

```python
# Pandas - изменяемый (mutable)
df['new_col'] = 1  # Изменяет df на месте

# Datasets - неизменяемый (immutable)
ds = ds.map(...)   # Возвращает НОВЫЙ объект
```

#### **Обработка пакетами:**

```python
# Pandas - построчно быстрее для малых данных
df.apply(func)

# Datasets - пакетная обработка ОБЯЗАТЕЛЬНА для скорости
ds.map(func, batched=True, batch_size=1000)
```

### 🎓 **9. Когда использовать что?**

**Используйте Pandas DataFrame когда:**
- ✅ Данные помещаются в память
- ✅ Нужна интерактивная работа в Jupyter
- ✅ Требуется гибкая индексация (loc, iloc)
- ✅ Работаете с табличными бизнес-данными
- ✅ Нужны агрегации и группировки (groupby)

**Используйте Datasets Dataset когда:**
- ✅ Работаете с большими данными (>RAM)
- ✅ Обучаете NLP/ML модели
- ✅ Нужно кэширование операций
- ✅ Требуется интеграция с Transformers
- ✅ Работаете с текстовыми данными
- ✅ Нужна параллельная обработка

### 🔄 **10. Конвертация между форматами**

```python
# DataFrame → Dataset
from datasets import Dataset
ds = Dataset.from_pandas(df)

# Dataset → DataFrame
df = ds.to_pandas()

# Dataset с pandas форматом (без конвертации)
ds_pandas = ds.with_format('pandas')
# Теперь ds_pandas[0] вернет pd.DataFrame
```

### 💡 **11. Практический пример: гибридный подход**

```python
from datasets import load_dataset
import pandas as pd

# Загружаем большой датасет через Datasets
ds = load_dataset('csv', data_files='huge_file.csv')

# Обрабатываем эффективно через Datasets
ds = ds.map(lambda x: {'processed': preprocess(x['text'])}, batched=True)

# Конвертируем в Pandas для анализа
df = ds.to_pandas()

# Используем мощь Pandas для EDA
print(df.describe())
print(df.groupby('category').mean())

# Обратно в Dataset для обучения модели
ds_final = Dataset.from_pandas(df)
```

---

## 📌 Резюме

| Критерий | Pandas | Datasets |
|----------|--------|----------|
| **Размер данных** | Малые/средние | Любые (особенно большие) |
| **Скорость (малые данные)** | ⚡ Быстрее | Медленнее |
| **Скорость (большие данные)** | Медленнее/невозможно | ⚡ Быстрее |
| **Память** | Вся в RAM | Memory mapping |
| **Кэширование** | ❌ Нет | ✅ Автоматическое |
| **ML интеграция** | Ручная | ✅ Нативная |
| **Гибкость** | ✅ Очень высокая | Средняя |
| **Кривая обучения** | Пологая | Средняя |

**Лучшая практика:** Используйте Datasets для загрузки и предобработки больших данных, затем конвертируйте в Pandas для детального анализа и визуализации, и обратно в Datasets для обучения моделей.
