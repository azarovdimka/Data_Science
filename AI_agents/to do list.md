# нужно создать файл с описанием всех технологий и методов, которые использовались, чтобы обозначить суть и приемущества проекта.

# 1) Нужно проверить по всей базе дубликаты при помощи MinHash
```
from datasketch import MinHash, MinHashLSH

def clean_text(text: str) -> str:
    """Базовая очистка: удаление множественных пробелов"""
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def get_minhash(text: str, num_perm: int = 128) -> MinHash:
    """Создание MinHash отпечатка для текста"""
    m = MinHash(num_perm=num_perm)
    # Разбиваем на шинглы (3-граммы символов)
    for i in range(len(text) - 3):
        m.update(text[i:i+3].encode('utf8'))
    return m

## Пример: поиск дубликатов в небольшом наборе
sample_docs = [
    "RAG системы требуют качественной подготовки данных для работы.",
    "RAG системы требуют очень качественной подготовки данных для работы!!!", # Почти дубль
    "Vision Language Models (VLM) меняют подход к обработке документов.",
    "Сегодня хорошая погода в Москве.",
    "RAG системы нуждаются в качественной подготовке данных." # Еще один вариант
]

## Создаём LSH индекс
lsh = MinHashLSH(threshold=0.7, num_perm=128)  # threshold=0.7 означает 70% похожести

for i, doc in enumerate(sample_docs):
    minhash = get_minhash(clean_text(doc))
    lsh.insert(f"doc_{i}", minhash)

## Проверяем каждый документ на дубликаты
print("Поиск дубликатов:\n")
for i, doc in enumerate(sample_docs):
    query_minhash = get_minhash(clean_text(doc))
    duplicates = lsh.query(query_minhash)

    if len(duplicates) > 1:  # Больше одного = есть дубли
        print(f"Doc {i}: '{doc}...'")
        print(f"  └─ Похожие документы: {duplicates}\n")
```

## Поиск дубликатов:
```
Doc 0: 'RAG системы требуют качественной подготовки данных для работы....'
  └─ Похожие документы: ['doc_0', 'doc_1']

Doc 1: 'RAG системы требуют очень качественной подготовки данных для работы!!!...'
  └─ Похожие документы: ['doc_0', 'doc_1']
```

```
# Применяем дедупликацию к нашему датасету
def find_duplicates(df: pd.DataFrame, threshold: float = 0.8) -> List[List[int]]:
    """Находит группы дубликатов в датасете"""
    lsh = MinHashLSH(threshold=threshold, num_perm=128)

    # Индексируем все документы
    for idx, text in enumerate(df['text'].fillna('')):
        minhash = get_minhash(clean_text(text))
        lsh.insert(f"doc_{idx}", minhash)

    # Находим дубликаты
    seen = set()
    duplicate_groups = []

    for idx, text in enumerate(df['text'].fillna('')):
        if idx in seen:
            continue

        minhash = get_minhash(clean_text(text))
        duplicates = lsh.query(minhash)

        if len(duplicates) > 1:
            group = [int(d.split('_')[1]) for d in duplicates]
            duplicate_groups.append(group)
            seen.update(group)

    return duplicate_groups

# Поиск дубликатов
duplicate_groups = find_duplicates(df, threshold=0.85)

print(f"Найдено {len(duplicate_groups)} групп дубликатов")

if duplicate_groups:
    print(f"\nПример дубликатов:")
    for i, group in enumerate(duplicate_groups):
        print(f"\nДубликаты из группы {i + 1}")
        for idx in group:
            print(f"  - [{idx}]: {df.iloc[idx]['text']}")

```

# 2) Использовать VLM вместо PDF

**Преимущества VLM:**
- **Holistic understanding**: понимает визуальную структуру + текст + семантику одновременно
- **No OCR needed**: обрабатывает документ как изображение
- **Better on complex layouts**: таблицы, формы, диаграммы
- **Multimodal retrieval**: на **12% точнее** даже "идеального" OCR (NDCG@5)
- **Production-ready**: ColPali, ColQwen достигли промышленного качества

- **ColQwen-Omni** - все модальности сразу

# 3) В коде использовать тайпинг: def func(text: str) -> var:

# 4) для разбиения и парсинга текстов использовать принцип  RecursiveChunker

Метод разбивает документ по иерархическим уровням (заголовки → параграфы → предложения → токены).

**Плюсы:** сохраняет структуру и семантические границы.  
**Минусы:** требует правил для разных форматов и чуть сложнее поддерживать.

**Используйте для:** технических доков, markdown/HTML, книг и любых структурированных текстов.

[Документация RecursiveChunker](https://docs.chonkie.ai/python-sdk/chunkers/recursive-chunker)


# 5) использовать обогащение данных Metadata Enrichment

### Зачем нужно обогащение?

"Слепой" чанкинг часто теряет контекст:
- Кто автор этой информации?
- О какой дате идёт речь?
- Какие организации упомянуты?
- Где происходит действие?

Извлекая **метаданные**, мы можем:
1. Фильтровать результаты поиска ("только посты про OpenAI")
2. Улучшать релевантность (временная релевантность)
3. Строить графы знаний
4. Группировать документы по темам/авторам

### GLiNER — современный NER для русского языка

**GLiNER** (Generalist and Lightweight Named Entity Recognition) — это zero-shot NER модель:
- Не требует fine-tuning
- Можно задать любые лейблы "на лету"
- Хорошо работает с русским языком
- Легковесная (можно запустить в Colab)

[GLiNER на HuggingFace](https://huggingface.co/urchade/gliner_medium-v2.1)

```
from gliner import GLiNER

# Загружаем модель (компактная версия для Colab)
print("Загружаем GLiNER модель...")
model_ner = GLiNER.from_pretrained("urchade/gliner_medium-v2.1")

# Пример текста
text_for_ner = """
В феврале 2026 года OpenAI представила новую модель GPT-5 в Сан-Франциско.
Сэм Альтман заявил, что это прорыв в области искусственного интеллекта.
Компания планирует развернуть сервис в России и Китае до конца года.
"""

# Задаём лейблы (можно любые!)
labels = ["person", "organization", "date", "location", "product", "event"]

# Извлекаем сущности
entities = model_ner.predict_entities(text_for_ner, labels)

print("\n🔍 Найденные сущности:\n")
for entity in entities:
    print(f"  {entity['label']:15} → {entity['text']}")

```

# Нужно добавить удаление мусора и номрализация аббревиатур

