#!/usr/bin/env python3.9
"""
Настройка логирования для бота
"""
import logging
import urllib3
from config import LOG_FILE

# Отключаем логи для urllib3
urllib3.disable_warnings()
logging.getLogger('urllib3.connectionpool').setLevel(logging.WARNING)


def setup_logging():
    """Настройка системы логирования"""
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(LOG_FILE, encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    
    # Настройка логгера для telebot
    import telebot
    telebot.logger.setLevel(logging.INFO)
    
    return logging.getLogger(__name__)