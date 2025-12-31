# Система автоматической оценки экзаменов по русскому языку

## Описание проекта

Веб-приложение для автоматической оценки устных ответов на экзамене по русскому языку с использованием машинного обучения и анализа транскрибированной речи.


**Метрики проекта:**
train best_score 0.22
MAE (test): 0.22
Accuracy (test): 0.78
f1_score: 0.76

## Функциональность

1) Загрузка CSV файлов с данными экзаменов: ограничение на загрузку файлов 50 MB

2) Автоматическая оценка ответов на основе ML модели
- Веб-интерфейс для загрузки и скачивания файлов
- Адаптивный дизайн для desktop и mobile
- API для интеграции с внешними системами

## Критерии оценки

1. Ошибки в отдельных словах не считаются критичными
2. Акцент не влияет на оценку при понятной речи
3. Выполнение коммуникативной задачи
4. Использование полных предложений

## Диапазоны оценок

- Вопрос 1: 0-1 балл
- Вопрос 2: 0-2 балла  
- Вопрос 3: 0-1 балл
- Вопрос 4: 0-2 балла

## Расположение ключевых файлов

**Cайт проекта** http://vds.spb.su:8000 или ip: 80.78.254.33
**Заглавная страница** проекта лежит - \rus_exam_site\frontend\templates\index.html
**Тетрадка JupyterNotebook** с расчетами лежит в корне - rus_exam.ipynb
**Сама модель** в корне - rus_exam_site\evaluate_exam_model.pkl
**Файл с описанием** лежит в корне readme.md


## Структура проекта

```
├── backend/                      # Backend-часть приложения 
│   ├── main.py                   # Основной исполняемый файл приложения с API endpoints
│   └── handlers/                 # Обработчики HTTP-запросов
|   |   └── csv_handler.py        # ==> обработчик для CSV файлов: прием csv, предобрбаботка, передача в модель, возврат нового файла
|   |   └── exam_handler.py       # обработчик для экзаменов
|   |   └── stats_handler.py      # обработчик для статистики
|   |
|   └── handlers/                 # Обработчики 
|   |   └── csv_handler.py        # 
|   |   └── stats_handler.py      #
|   |
│   ├── services/                 # Бизнес-логика
|   |   └─ data_preprocessor.py   # Скачивает аудио, замеряет длину, морфологический анализ, создает новые признаки (столбцы)
|   |   └─ ml_service.py          # Сервис машинного обучения для оценки ответа 
|   |
│   └── utils/                    # Вспомогательные функции
|
├── config/                  # Конфигурационные файлы
|   └─settings.py            # Конфигурационные настройки
|
├── docs/                    # Документация проекта
|   └─ECONOMIC_ANALYSIS.md   # Экономическое обоснование проекта
|   └─PRESENTATION.md        # Презентация проекта
|   └─TECHNICAL_ANALYSIS.md  # Технический анализ
|
├── frontend/                # Frontend часть приложения
│   └─ static/               # Статические файлы (CSS, JS)
|   |  ├─ scc/               # Конфигурационные настройки
|   |  |  └─ style.scc       # Адаптивные стили 
|   |  ├─ images/            # Картинки
|   |  └─ js/                # JavaScript функциональность
|   |     └─ main.js         # Основной сценарий JavaScript
|   |
│   ├── templates/           # HTML шаблоны
|   |   └─ index.html        # главная страница с Bootstrap 5
│   └── components/          # Компоненты интерфейса
|
|
├── tests/                   # Тесты
└── evaluate_exam_model.pkl  # Сама модель
└── requirements.txt         # Зависимости Python
└── rus_exam_site.conf       # копия файла conf, который должен быть скопирован на сервер в директорию `/etc/nginx/conf.d/rus_exam_site.conf`
└── rus_exam_site.service    # копия файла service, который должен быть скопирован на сервер в директорию `/etc/systemd/system/rus_exam_site.service`
└── rus_exam.ipynb           # Рабочий джупитр ноутбук для анализа входящих данных, отладки алгоритмов и обучения модели
```

## Технологический стек

- **Backend**: FastAPI, Python 3.10+
- **Frontend**: HTML5, CSS3, JavaScript, Bootstra
- **ML**: scikit-learn, transformers, torch, TD-IDF, pymorphy3
- **Информация о скачанных аудио**: mutagen3


## Установка и запуск

# 1. Создать виртуальное окружение
python3.10.12 -m venv /opt/rus_exam_venv
source /opt/rus_exam_venv/bin/activate

# 2. Перейти в папку проекта на сервере и скопирвоать туда весь проект
cd /var/www/rus_exam_site

# 3. Установить все зависимости
pip install -r requirements.txt

# 4. Проверить ключевые библиотеки
python -c "import sklearn; print(sklearn.__version__)"

# 5. Запустите приложение: `python backend/main.py`



**Соержание файла .service**

Запуск приложения как сервис:
Создать /etc/systemd/system/rus_exam_site.service:

```
[Unit]
Description=Russian Exam Site
After=network.target

[Service]
User=www-data
WorkingDirectory=/var/www/rus_exam_site
ExecStart=/usr/bin/python3 backend/main.py
Restart=always

[Install]
WantedBy=multi-user.target


sudo systemctl enable rus_exam_site
sudo systemctl start rus_exam_site
```

**создание конфигурации ngnix**

/etc/nginx/sites-available/rus_exam_site/rus_exam_site.conf
```
server {
    listen 80;
    server_name vds.spb.su;
    
    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
    
    location /static/ {
        alias /var/www/rus_exam_site/frontend/static/;
    }
}
```

systemctl reload nginx


**Запуск сервера:** 
- локально в терминале: python backend/main.py или docker-compose up
- открыть в браузере: http://localhost:8000

