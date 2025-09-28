#!/usr/bin/env python3.9
"""
Конфигурация бота
"""
import os
from dotenv import load_dotenv

# Загрузка переменных окружения
load_dotenv()

# Токены и API ключи
TELEGRAM_BOT_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN')
YANDEX_API_KEY = os.getenv('YANDEX_API_KEY')

# ID администраторов
ADMIN_ID = 157758328

# Список пользователей KRS
KRS_LIST = [
    157758328,  # - azarov
    5259596384,  # - nemchinova
    595984290,  # - Толоконникова
    5275895896,  # Баскаков Сергей
    5006193045,  # - alekseev
    6413267438,  # - urban
    1910564254,  # - urban
    5181039257,  # - lapina
    5023870980  # - klimova
]

# Настройки логирования
LOG_FILE = 'flight_attendant_bot.log'
CONTEXT_LOG_FILE = "context_responses.log"

# Настройки YandexGPT
YANDEX_GPT_URL = "https://llm.api.cloud.yandex.net/foundationModels/v1/completion"
YANDEX_TOKENIZER_URL = "https://llm.api.cloud.yandex.net/foundationModels/v1/tokenizeCompletion"
MAX_TOKENS = 30000