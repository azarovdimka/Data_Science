#!/usr/bin/env python3.9
"""
Модуль для работы с хранилищем данных
"""
import logging
from datetime import datetime
from yandex_gpt import YandexGPT
from config import CONTEXT_LOG_FILE

logger = logging.getLogger(__name__)


class DataStore:
    """DataStore: только поиск и логирование контекста, НЕ добавление нового текста в базу данных"""
    
    def __init__(self):
        self.yandex_gpt = YandexGPT()
        self.context_log_file = CONTEXT_LOG_FILE
    
    def log_context_and_response(self, user_id, question, context_fragments, response):
        """Логирование контекста и ответа"""
        try:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            # Запись в файл
            with open(self.context_log_file, 'a', encoding='utf-8') as f:
                f.write("=" * 80 + "\n")
                f.write(f"TIMESTAMP: {timestamp}\n")
                f.write(f"USER_ID: {user_id}\n")
                f.write(f"QUESTION: {question}\n")
                f.write(f"FRAGMENTS_COUNT: {len(context_fragments)}\n")
                f.write(f"TOTAL_CONTEXT_CHARS: {sum(len(fragment) for fragment in context_fragments)}\n")
                f.write(f"RESPONSE_CHARS: {len(response)}\n")
                f.write("\n--- CONTEXT FRAGMENTS ---\n")
                
                for i, fragment in enumerate(context_fragments, 1):
                    f.write(f"\n[FRAGMENT {i}] ({len(fragment)} chars):\n")
                    f.write(fragment)
                    f.write("\n")
                
                f.write("\n--- GENERATED RESPONSE ---\n")
                f.write(response)
                f.write("\n\n")
            
            logger.info(f"Context and response logged for user {user_id}")
            
        except Exception as e:
            logger.error(f"Error logging context and response: {str(e)}")
    
    def search_and_generate_response(self, question, user_id):
        """Поиск в базе данных и генерация ответа"""
        try:
            # Здесь должен быть код поиска в базе данных
            # Пока заглушка, так как полный код не виден
            context_fragments = []  # Результат поиска в БД
            
            if not context_fragments:
                return "Извините, я не нашел информации по вашему вопросу."
            
            # Объединяем найденные фрагменты в контекст
            context = "\n\n".join(context_fragments)
            
            # Генерируем ответ через YandexGPT
            response = self.yandex_gpt.generate_response(question, context)
            
            # Логируем контекст и ответ
            self.log_context_and_response(user_id, question, context_fragments, response)
            
            return response
            
        except Exception as e:
            logger.error(f"Error in search_and_generate_response: {str(e)}")
            return "Произошла ошибка при обработке вашего запроса."