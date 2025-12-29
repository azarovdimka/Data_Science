#!/usr/bin/env python3.9
"""
Модуль для работы с YandexGPT API
"""
import logging
import requests
from config import YANDEX_API_KEY, YANDEX_GPT_URL, YANDEX_TOKENIZER_URL, MAX_TOKENS

logger = logging.getLogger(__name__)


class YandexGPT:
    def __init__(self, api_key=None):
        self.api_key = api_key or YANDEX_API_KEY
        self.url = YANDEX_GPT_URL
        self.tokenizer_url = YANDEX_TOKENIZER_URL
        self.headers = {
            "Authorization": f"Api-Key {self.api_key}",
            "Content-Type": "application/json"
        }
        self.max_tokens = MAX_TOKENS

    def count_tokens(self, messages):
        """Подсчет токенов в сообщениях"""
        data = {
            "modelUri": "gpt://b1gitmbk3b5pv7ve1vtv/yandexgpt-lite",
            "messages": messages
        }
        
        try:
            response = requests.post(self.tokenizer_url, headers=self.headers, json=data)
            response.raise_for_status()
            result = response.json()
            return result.get('tokens', [])
        except Exception as e:
            logger.warning(f"Failed to count tokens: {e}")
            # Примерная оценка: 1 токен ≈ 4 символа для русского текста
            total_chars = sum(len(msg.get('text', '')) for msg in messages)
            return [{'token': 'estimated'}] * (total_chars // 4)
    
    def truncate_context(self, context, prompt, max_context_tokens=25000):
        """Обрезает контекст до допустимого размера"""
        messages = [
            {"role": "system", "text": context},
            {"role": "user", "text": prompt}
        ]
        
        tokens = self.count_tokens(messages)
        token_count = len(tokens)
        
        if token_count <= self.max_tokens:
            return context
        
        # Обрезаем контекст пропорционально
        context_lines = context.split('\n')
        target_lines = int(len(context_lines) * max_context_tokens / token_count)
        truncated_context = '\n'.join(context_lines[:target_lines])
        
        logger.warning(f"Context truncated from {token_count} to ~{target_lines} lines")
        return truncated_context

    def generate_response(self, prompt, context):
        """Генерация ответа через YandexGPT"""
        # Проверяем размер перед отправкой
        context = self.truncate_context(context, prompt)
        
        data = {
            "modelUri": "gpt://b1gitmbk3b5pv7ve1vtv/yandexgpt-lite",
            "completionOptions": {
                "stream": False,
                "temperature": 0.2,
                "maxTokens": 2000
            },
            "messages": [
                {"role": "system", "text": context},
                {"role": "user", "text": prompt}
            ]
        }

        try:
            logger.info(f"Sending request to YandexGPT API with prompt: {prompt[:100]}...")
            response = requests.post(self.url, headers=self.headers, json=data)

            logger.info(f"Response status code: {response.status_code}")
            logger.info(f"Response content: {response.text[:200]}")

            response.raise_for_status()
            result = response.json()

            # Проверяем структуру ответа
            if 'result' in result and 'alternatives' in result['result'] and result['result']['alternatives']:
                return result['result']['alternatives'][0]['message']['text']
            else:
                logger.error(f"Unexpected response structure: {result}")
                return "Извините, произошла ошибка при обработке ответа"

        except requests.exceptions.HTTPError as http_err:
            logger.error(f"HTTP error occurred: {http_err}")
            logger.error(f"Response content: {response.text}")
            raise
        except Exception as e:
            logger.error(f"Error in YandexGPT generation: {str(e)}")
            raise