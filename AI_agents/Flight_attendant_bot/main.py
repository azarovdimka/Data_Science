#!/usr/bin/env python3.9
"""
Главный файл запуска бота
"""
import sys
import telebot
from logger_config import setup_logging
from config import TELEGRAM_BOT_TOKEN, ADMIN_ID
from bot_handlers import BotHandlers
import settings as s

# Настройка логирования
logger = setup_logging()

def main():
    """Главная функция запуска бота"""
    try:
        # Инициализация бота
        bot = telebot.TeleBot(TELEGRAM_BOT_TOKEN)
        logger.info("Bot initialized successfully")
        
        # Настройка обработчиков
        handlers = BotHandlers(bot)
        logger.info("Handlers setup completed")
        
        # Уведомление о запуске
        try:
            bot.send_message(ADMIN_ID, "Бот перезапущен")
        except Exception as e:
            logger.warning(f"Failed to send startup message to admin: {e}")
        
        # Запуск бота
        logger.info("Starting bot polling...")
        bot.polling(none_stop=True, interval=0, timeout=20)
        
    except Exception as e:
        logger.critical(f"Failed to initialize bot: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()