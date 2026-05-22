#!/usr/bin/env python3.9

import os
import re
import sys
from datetime import datetime
import json

import urllib3
from dotenv import load_dotenv
import logging
import requests
import PyPDF2
from langchain.chains import RetrievalQA
import telebot  # чтобы работал telebot - удалить telebot, и установить Pytelegrambotapi, написанным оставить telebot
import baza
from telebot import types
from random import choice
import exception_logger
import handler_db
import threading
import time
import traceback
import settings as s

# Отключаем логи для urllib3
urllib3.disable_warnings()
logging.getLogger('urllib3.connectionpool').setLevel(logging.WARNING)

# Настройка логирования
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('flight_attendant_bot.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# # Загрузка переменных окружения из файла .env
load_dotenv()

try:
    bot = telebot.TeleBot(os.getenv('TELEGRAM_BOT_TOKEN'))  # , state_storage=STATE_STORAGE)
except Exception as e:
    logger.critical(f"Failed to initialize bot: {str(e)}")
    sys.exit(1)

# Настройка логгера для telebot
telebot.logger.setLevel(logging.INFO)

bot.send_message(s.ADMIN_ID, "Бот перезапущен")

admin_id = 157758328

krs_list = [
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

class YandexGPT:
    def __init__(self, api_key):
        self.api_key = api_key
        self.url = "https://llm.api.cloud.yandex.net/foundationModels/v1/completion"
        self.tokenizer_url = "https://llm.api.cloud.yandex.net/foundationModels/v1/tokenizeCompletion"
        self.headers = {
            "Authorization": f"Api-Key {self.api_key}",
            "Content-Type": "application/json"
        }
        self.max_tokens = 30000  # Оставляем запас от лимита 32768

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

            # Добавим логирование ответа для отладки
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


def process_pdf_text(text: str):
    """Обработка и добавление текста из PDF в базу данных"""
    try:
        # Очистка текста от лишних пробелов
        cleaned_text = ' '.join(text.split())
        
        # Разделение текста на части
        parts = re.split(r'(?=\b\d+(?:\.\d+)*\.|Глава|-----)', cleaned_text)
        logger.info(f"Text split into {len(parts)} parts")
        
        success_count = 0
        for i, part in enumerate(parts):
            # Очистка каждой части
            part = re.sub(r'(?:Рис\.\s\d+\s+.*)', '', part)
            part = re.sub(r'Издание:\s*\d+;\s*Изменение:\s*\d+', '', part)
            part = re.sub(r'Стр\.\s[A-ZА-Я]/\d-\s\d+', '', part)
            part = re.sub(r' РКЭ Часть | п.\d+. ', '', part)
            part = re.sub(r'\d\sп\.', '', part)
            part = re.sub(r'РД-ГД-\d{2}-\d{2}\s+Стр\.\s[А-Я]\s/\d-\s\d', '', part)
            part = re.sub(r'-----', '', part)
            part = re.sub(r'Россия', '', part)
            
            # Убираем лишние пробелы и проверяем на пустоту
            part = part.strip()
            if len(part) < 10:  # Пропускаем слишком короткие фрагменты
                logger.debug(f"Skipping short part {i+1}: '{part[:50]}...'")
                continue
            
            result = handler_db.add_text(part)
            if result:
                success_count += 1
                logger.info(f"Part {i+1} added to database ({len(part)} chars)")
            else:
                logger.error(f"Failed to add part {i+1} to database")
        
        logger.info(f"Successfully added {success_count} parts to database")
        return success_count > 0

    except Exception as e:
        logger.error(f"Error processing PDF text: {str(e)}")
        return None


# Хранилище данных
class DataStore:
    """DataStore: только поиск и логирование контекста, НЕ добавление нового текста в базу данных"""
    def __init__(self):
        self.yandex_gpt = YandexGPT(os.getenv('YANDEX_API_KEY'))
        # self.database_file = "knowledge_base.txt"
        self.context_log_file = "context_responses.log"
    
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
            logger.error(f"Failed to log context and response: {e}")



            logger.error(f"Error adding text to database: {str(e)}")
            return None


    def get_relevant_context(self, question: str, max_chars: int = 8000) -> str:
        """Поиск релевантного контекста с ограничением размера"""
        question_words = {word.lower() for word in question.split() if len(word) > 2}
        all_texts = handler_db.get_all_texts()
        
        # Особая логика для поиска номеров телефонов
        question_lower = question.lower()
        is_phone_search = any(word in question_lower for word in ['телефон', 'телефона', 'позвонить', 'звонить', 'вызвать', 'номер', '+7', '8('])
        if is_phone_search:
            max_chars = 20000  # Увеличиваем лимит для поиска телефонов
            logger.info("Phone search detected, increasing context limit")
        
        # Оптимизированный генератор ключевых фраз
        key_phrases = []
        important_words = [word for word in question_words if len(word) > 3][:5]  # Макс 5 слов
        
        # Генерируем только самые важные пары
        if len(important_words) >= 2:
            # Берем только первые 3 пары (макс 6 фраз)
            from itertools import combinations
            pairs = list(combinations(important_words, 2))[:3]
            for word1, word2 in pairs:
                key_phrases.extend([f"{word1} {word2}", f"{word2} {word1}"])
        
        # Исключаем стоп-слова
        stop_words = {'котор', 'в', 'во', 'на', 'с', 'со', 'за', 'из', 'о', 'об', 'от', 'по', 
                     'для', 'как', 'что', 'это', 'или', 'при', 'все', 'его', 'её', 'их'}
        question_words = question_words - stop_words
        
        scored_texts = []
        for text in all_texts:
            text_content = str(text[1])
            if len(text_content) < 50:  # Пропускаем слишком короткие тексты
                continue
                
            text_words = {word.lower() for word in text_content.split() if len(word) > 2}
            text_words = text_words - stop_words
            
            # Находим совпадения
            matching_words = question_words & text_words
            
            if matching_words:
                # Базовая оценка по словам
                word_score = len(matching_words) * 100 / len(text_content)
                
                # Бонус за точные фразы и близкое расположение слов
                phrase_bonus = 0
                text_lower = text_content.lower()
                
                # Проверяем точные фразы
                for phrase in key_phrases:
                    if phrase in text_lower:
                        phrase_bonus += 1000  # Очень высокий бонус
                
                # Оптимизированный бонус за близкость (только для первых 2 слов)
                proximity_bonus = 0
                if len(important_words) >= 2:
                    word1, word2 = important_words[0], important_words[1]
                    if word1 in text_lower and word2 in text_lower:
                        pos1 = text_lower.find(word1)
                        pos2 = text_lower.find(word2)
                        distance = abs(pos1 - pos2)
                        if distance < 100:
                            proximity_bonus = max(0, 200 - distance * 2)
                
                phrase_bonus += proximity_bonus
                
                # Бонус за номера телефонов (паттерны)
                phone_bonus = 0
                if is_phone_search:
                    phone_patterns = [r'\+7[\d\s\(\)\-]{10,}', r'8[\d\s\(\)\-]{10,}', r'\d{3}[\-\s]\d{2}[\-\s]\d{2}']
                    for pattern in phone_patterns:
                        if re.search(pattern, text_content):
                            phone_bonus += 500
                
                total_score = word_score + phrase_bonus + phone_bonus
                scored_texts.append((text_content, total_score, len(matching_words), phrase_bonus))
        
        # Сортируем по релевантности (сначала по общему скору, потом по количеству слов)
        scored_texts.sort(key=lambda x: (x[1], x[2]), reverse=True)
        
        if key_phrases:
            logger.info(f"Key phrases detected: {key_phrases}")
        logger.info(f"Found {len(scored_texts)} relevant texts, top 5 scores: {[x[1] for x in scored_texts[:5]]}")
        
        # Собираем контекст с ограничением по размеру
        result_texts = []
        total_chars = 0
        
        for text_content, score, matches, phrase_bonus in scored_texts:
            if total_chars + len(text_content) > max_chars:
                # Обрезаем последний текст если нужно
                remaining = max_chars - total_chars
                if remaining > 200:  # Добавляем только если остается достаточно места
                    result_texts.append(text_content[:remaining] + "...")
                break
            
            result_texts.append(text_content)
            total_chars += len(text_content)
            
            # Динамический лимит фрагментов
            max_fragments = 50 if is_phone_search else 15
            if len(result_texts) >= max_fragments:
                break
        
        context = "\n\n".join(result_texts)
        logger.info(f"Selected {len(result_texts)} relevant fragments, total chars: {len(context)}")
        if key_phrases and result_texts:
            logger.info(f"Top fragment score breakdown: word_score + phrase_bonus + phone_bonus = total")
        
        # Сохраняем фрагменты для логирования
        self.last_context_fragments = result_texts
        
        return context

    def get_answer(self, question: str, user_id: int = None) -> str:

        question = re.sub('[:,.!?()-/"]', ' ', question)
        context = self.get_relevant_context(question)

        if not context:
            return "К сожалению, в базе знаний нет информации для ответа на этот вопрос."

        prompt = (f"На основе предоставленного контекста ответьте на вопрос. "
                  f"Ответ должен быть структурированным, развернутым, полным и точным. "
                  f"Если ты упоминаешь в ответе аббревиатуру СБ, то запомни, что эта аббревиатура "
                  f"расшифровывается только как старший бортпроводник, никакой службы безопасности нет. Аббревиатура ОК - расшифровывается как обслуживающая компания. "
                  f"\n\nКонтекст:\n{context}\n\nВопрос: {question}")
        
        # Обрезаем контекст если он слишком большой
        truncated_context = self.yandex_gpt.truncate_context(context, prompt)
        if truncated_context != context:
            prompt = prompt.replace(context, truncated_context)
            logger.info(f"Context: {context[:100]}\n"
                        f" len {len(context)} symbols truncated to {len(truncated_context)} symbols")
        
        response = self.yandex_gpt.generate_response(prompt, truncated_context)
        
        # Логируем контекст и ответ
        if hasattr(self, 'last_context_fragments') and user_id:
            self.log_context_and_response(user_id, question, self.last_context_fragments, response)
        
        return response


    def forget_context(self, text_to_forget: str) -> bool:
        """Удаление определенного контекста из базы данных"""
        try:
            # Удаление из базы данных
            result = handler_db.delete_text(text=text_to_forget)

            if result:
                bot.send_message(admin_id, f'Я забыл про {text_to_forget}')
                logger.info(f"Context successfully deleted from database")
                return True
            else:
                bot.send_message(admin_id, f'Неуспешно.')
                logger.error(f"Failed to delete context from database")
                return False

        except Exception as e:
            logger.error(f"Error deleting context from database: {str(e)}")
            bot.send_message(admin_id, f'Неуспешно. {str(e)}')
            return False


# Инициализация хранилища данных
data_store = DataStore()

## -*- coding: utf8 -*-


def service_notification(message):
    """Уведомление на случай проведения технических работ на сервере."""
    bot.send_message(message.chat.id, 'На сервере проводятся технические работы. Возможна некорректная работа '
                                      'телеграм-бота. Это продлится недолго. Приносим свои извинения за доставленные '
                                      'неудобства. Если что-то не получится - попробуйте завтра.')
    bot.send_message(157758328, f"Отправлено уведомление о некорректной работе телеграм-бота.")


@bot.message_handler(content_types=['photo'])
def handle_docs_photo(message):
    """пересылает разработчику картинку отправленную пользователем. Сделано для верификации по айдишке"""
    bot.send_photo(157758328, message.photo[0].file_id)
    new_photo_notification = "Пользователь {0.first_name} {0.last_name} @{0.username} id {0.id} прислал " \
                             "фото.".format(message.from_user, message.from_user, message.from_user,
                                            message.from_user)
    bot.send_message(157758328, new_photo_notification)
    bot.send_message(message.chat.id, 'Бот не работает.')
    bot.send_message(message.chat.id,
                     "Фото отправлено успешно. Пожалуйста, ожидайте, о результате мы Вам сообщим. Ожидание может продлиться до суток.")


@bot.message_handler(commands=['start'])
def welcome(message):
    """При первом подключении пользователя к боту - выводит приветственный стикер, приветственную речь. Также в этой
    функции обозначены кнопки, которые будут всегда отображаться под полем ввода запроса."""

    logger.info(f"User {message.from_user.id} started the bot")

    with open('static/AnimatedSticker.tgs', 'rb') as sti:
        bot.send_sticker(message.chat.id, sti)

    if handler_db.check_access(message.chat.id):
        name = handler_db.get_name_surname(message.chat.id).split()[0]
        # bot.send_message(message.chat.id, permission_message, reply_markup=select_action())
    else:
        bot.send_message(message.chat.id,
                         # f"Представьтесь, пожалуйста. Напишите свой табельный номер, фамилию, имя через пробел.
                         # Например: \n123456 Смирнов Иван")
                         "Доступ для вас ограничен.")
        logger.info(f"Сообщили {message.from_user.id} о том, что у него нет доступа к боту.")
        return

    bot.reply_to(message,  # отвечает цитированием
                 f"Здравствуйте, {name}! Я бот, который постарается ответить на ваши вопросы.\n"
                 "\tЕсли не удалось получить ответ на вопрос - постарайтесь его переформулировать "
                 "вопрос или задать его подробнее. При формулировке вопроса старайтесь обращать внимание на использование "
                 "ключевых уникальных слов в вопросе, которые должны помочь найти именно этот ответ, можете добавить слова, "
                 "которые точно должны быть в ответе... При ведении диалога учитывайте, что он не помнит предыдущий контекст, "
                 "поэтому вопрос придется еще раз написать по-другому. Если после этого все "
                 "равно не удалось получить ответ - отправьте в бота pdf-файл, в котором содержится эта информация.")
    return


order_dict = {}

@bot.message_handler(commands=['donate'])
def faq(message):
    bot.send_message(message.chat.id, 'В авиакомпании я стремлюсь внедрять и развивать технологии искусственного интеллекта. Это моя личная инициатива, которая не приносит дохода.\n\n'
                                      'Однако я вкладываю в проект свои ресурсы: арендую сервер, оплачиваю хостинг и использую не бесплатные генеративные технологии. Кроме того, я трачу своё время, знания и усилия.\n\n'
                                      'Если вы хотите поддержать проект, вы можете сделать добровольное пожертвование на любую сумму. Для этого отправьте средства по номеру телефона 79992023315. Спасибо за вашу помощь!')
    logger.info(f"Предложили {message.from_user.id} донат.")
    return

@bot.message_handler(commands=['faq'])
def faq(message):
    bot.send_message(message.chat.id, '1. Чтобы получить ответ, просто задайте свой вопрос.\n\n'
                                      'Если бот не понял ваш вопрос, попробуйте переформулировать его или добавить слова, '
                                      'которые точно должны быть в ответе. Например, вместо "35 код задержки" скажите '
                                      '"код задержки 35" или "список кодов задержки". Вместо "инструктаж пассажира-помощника А319" '
                                      'напишите "текст инструктажа пассажира-помощника А319" или "инструктаж пассажира-помощника А319 '
                                      'эвакуация".\n\n'
                                      '2. Обучить бота можно двумя способами:\n\n'
                                      '- Начните сообщение со слова "запомни", затем напишите вопрос и ответ в одном сообщении. '
                                      'Например: "Запомни, есть ли жизнь на Марсе? Да, есть".\n'
                                      '- Отправьте PDF файл: наберите текст в Word, сохраните его как PDF и отправьте в бота. '
                                      'В одном файле можно указать много вопросов и ответов, главное — разделить их пятью дефисами (-----), '
                                      'словом "Глава" или цифрами через точку (например, 1.2.3.4).\n\n'
                                      'Общая рекомендация при обучении: указывайте все явно и подробно, не смешивайте разные '
                                      'темы и вопросы в одном тексте.\n\n'
                                      '3. Если бот дал вам неправильный ответ или ничего не ответил, но вы знаете правильный ответ '
                                      'или хотите добавить новую информацию, напишите в одном сообщении слово "запомни", '
                                      'затем свой вопрос и ответ через пробел.')
    logger.info(f"Рассказали FAQ пользователю {message.from_user.id}")
    return

@bot.message_handler(commands=['train'])
def train(message):

    bot.send_message(message.chat.id, 'Приглашаем вас помочь в обучении бота. Это требует времени, знаний и ресурсов.\n\n'
                                      'Обучить бота и добавить информацию можно двумя способами:\n\n'
                                      '1. Начните сообщение со слова "Запомни", затем введите текст, который нужно сохранить. '
                                      'Он должен быть литературным, грамотным и структурированным. Избегайте жаргона и сниженной лексики. '
                                      'Первая часть текста должна содержать развернутый вопрос с наибольшим количеством вопросительных '
                                      'слов и деталей, а вторая — подробный и последовательный ответ. Однако помните, что бот не любит '
                                      'лишние прилегающие символы (скобки, двоеточия, вопросительные и восклицательные знаки избегайте их, также не используется дробные подпункты при перечислении порядка действий 1.1, 1.2 и т.д. по этим маркерам он привык разбивать большие  главы документов на отдельные вопросы и на отдельные блоки, так что при перечислении порядка действий он не запомнит ничего, лучше просто 1. ... 2. ... или дефис), слишком длинные сообщения, поэтому ограничьтесь 850 символами. Пример: "Запомни: '
                                      'Есть ли жизнь на Марсе? Да, на Марсе есть жизнь.\n\n'
                                      '2. Отправьте PDF-файл. Наберите текст в Word, сохраните как PDF и отправьте боту. '
                                      'Затем задайте вопрос, чтобы проверить, успешно ли бот использует этот файл. '
                                      'В одном файле можно указать несколько вопросов и ответов, разделенных пятью дефисами. \n'
                                      'Например: \n первый блок текста с вопросом ответом...\n-----\nвторой блок с текстом и вопросом\n\n'
                                      'Рекомендуется указывать все явно и подробно, не смешивать разные темы и вопросы в одном блоке.')


    try:
        fio = handler_db.get_name_surname(message.chat.id)
        logger.info(f"User {message.from_user.id} {fio} requested how to train bot")


    except Exception as e:
        logger.error(f"Error in start_question: {e}\n traceback: {traceback.format_exc()}")
        bot.reply_to(message, "Произошла ошибка при установке режима вопросов")


@bot.message_handler(content_types=['document'])
def handle_pdf(message):

    if not handler_db.check_access(message.chat.id):
        bot.send_message(message.chat.id, "Доступ для вас ограничен.")
        logger.info(f"Сообщили, что доступ для {message.chat.id} ограничен")
        return


    try:
        logger.info(f"User {message.from_user.id} sent a PDF file")

        # Проверяем, является ли файл PDF
        if not message.document.mime_type == 'application/pdf':
            bot.reply_to(message, "Пожалуйста, отправьте PDF файл")

            return

        file_info = bot.get_file(message.document.file_id)
        downloaded_file = bot.download_file(file_info.file_path)

        # Сохраняем PDF временно
        with open('temp.pdf', 'wb') as temp_file:
            temp_file.write(downloaded_file)

        # Извлекаем текст из PDF
        text = ""
        with open('temp.pdf', 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            for page in pdf_reader.pages:
                text += page.extract_text()

        # Если текст успешно извлечен
        if text.strip():
            # Добавляем текст в базу данных
            result = process_pdf_text(text)
            if result:
                bot.reply_to(message, f"PDF файл успешно обработан и добавлен в базу знаний.")
                last_record = handler_db.get_last_data()
                if last_record:
                    bot.send_message(message.chat.id,
                                     f"Последняя запись: ID={last_record[0]}, Content='{last_record[1][:50]}...'")
                logger.info(f"Successfully processed PDF from user {message.chat.id}")
                return
        else:
            bot.reply_to(message, "Не удалось извлечь текст из PDF файла")
            logger.warning(f"No text extracted from PDF from user {message.from_user.id}")

    except Exception as e:
        logger.error(f"Error processing PDF: {str(e)}\n traceback: {traceback.format_exc()}")
        bot.reply_to(message, f"Произошла ошибка при обработке PDF файла {traceback.format_exc()}")
    finally:
        # Удаляем временный файл
        if os.path.exists('temp.pdf'):
            os.remove('temp.pdf')


@bot.message_handler(content_types=["text"])  #
def conversation(message):
    """Модуль для общения и взаимодействия с пользователем. Декоратор будет вызываться когда боту напишут текст."""

    if "добавить информацию" in message.text.lower():
        bot.send_message(message.chat.id, 'Чтобы добавить информацию, отправьте сообщение, которое начинается со слова запомни, а также содержит подробный вопрос и развернутый ответ, или же отправьте PDF файл, который нужно сохранить в базу знаний.')
        return

    user_id = message.chat.id
    name = handler_db.get_name_surname(message.chat.id).split()[0]
    fio = handler_db.get_name_surname(message.chat.id)

    logger.info(f"User {message.from_user.id} asked a question: {message.text}")

    bot.send_message(s.ADMIN_ID, f'пользователь {handler_db.get_tab_number(user_id)} {fio} спросил:\n  - {message.text}')


    def general_menu():
        """Основаня клавиатура внизу экрана"""
        general_menu = types.ReplyKeyboardMarkup(resize_keyboard=True, one_time_keyboard=False)
        btn5 = types.KeyboardButton('Добавить информацию')
        btn6 = types.KeyboardButton('Обратная связь')  # 'Заказ\nвыходных')
        general_menu.add(btn5, btn6)
        return general_menu

    def messaging_process(message):
        """При принудительном вызове функции рассылает всем сообщения со скоростью 1 человек в 3 секунды"""
        mess = message.text.split()
        counter_users = 0
        general_counter = 0
        for user_id in list_id:
            general_counter += 1
            user_id, surname, name, tab_number, password, messaging, check_permissions, night_notify, plan_notify, \
                autoconfirm, time_depart = handler_db.fetch_user_for_plan(user_id)
            fio = f'{user_id} {name} {surname}'
            if messaging:
                try:
                    bot.send_message(user_id, f'{name}, {" ".join(mess[2:])}', reply_markup=general_menu())
                    counter_users += 1
                    bot.send_message(157758328, f"Сообщение успешно отравлено {fio}")  # TODO временно
                    time.sleep(3)
                except Exception as exc:  # если случилась ошибка при отправке сообщений пользователю
                    exc_event = exception_logger.writer(exc=exc, request='рассылка сообщений пользователям',
                                                        fio=fio, answer='сообщение не удалось отправить ')
                    bot.send_message(157758328, exc_event)
                    bot.send_message(157758328,
                                     f"сообщение не удалось отправить {fio} ошибка {exc}.")  # TODO временно
        bot.send_message(157758328,
                         f"всего разослано {counter_users} чел. из {general_counter} чел.")  # TODO временно
        return

    if message.text in baza.greetings:
        bot.send_message(user_id, 'Привет! Буду рад тебе помочь, задавай свой вопрос.', reply_markup=general_menu())
        return

    if message.text.lower() in "обратная связь /feedback пригласить человека Написать разработчику автору азарову программисту справка как " \
                               "сообщить о проблеме ошибке неточности устаревшей информации работает неправильно /write":
        bot.send_message(157758328, f"Пишите сюда @AzarovML")
        return

    if "запомни" in message.text.lower():
        try:
            message.text = message.text.replace(':', '')
            text = message.text.split()[1:]  # Получаем текст, исключая первое слово (запомни)
            result = handler_db.add_text(' '.join(text))
            if result:
                bot.send_message(message.chat.id, f"Запомнил")
                last_record = handler_db.get_last_data()
                if last_record:
                    bot.send_message(message.chat.id, f"Последняя запись: ID={last_record[0]}, Content='{last_record[1][:50]}...'")
                return
            else:
                bot.send_message(message.chat.id, f"Что-то пошло не так. Попробуйте еще раз.")
        except Exception:
            bot.send_message(message.chat.id, f"Что-то пошло не так. Попробуйте еще раз.")
            bot.send_message(157758328, f"ОШИБКА: {traceback.format_exc()}")
            return

    if "забудь, что" in message.text.lower():
        try:
            text = message.text.split()[2:]
            result = data_store.forget_context(' '.join(text))
            if result:
                bot.send_message(user_id, f"Забыл успешно.")
                return
            else:
                bot.send_message(user_id, f"Забыть не удалось.")
                return
        except Exception:
            bot.send_message(user_id, f"Забыть не удалось.")
            bot.send_message(157758328, f"ОШИБКА: {traceback.format_exc()}")
            return

    if "выйти" in message.text.lower():
        bot.send_message(message.chat.id, "Хорошего дня! Если что - обращайтесь.", reply_markup=general_menu())
        return

    if "удалить дубликаты" in message.text.lower():
        deleted_count = handler_db.remove_duplicates()
        if deleted_count >= 0:
            bot.send_message(157758328, f"Удалено дубликатов: {deleted_count}")
        else:
            bot.send_message(157758328, f"Ошибка при удалении дубликатов")
        return

    if "написать по id" in message.text.lower():
        mess = message.text.split()
        try:
            bot.send_message(int(mess[3]), ' '.join(mess[4:]).capitalize(), reply_markup=general_menu())
            bot.send_message(157758328, "Сообщение пользователю отправлено успешно.")
        except Exception:
            bot.send_message(157758328, f"Пользователь не подключен к телеграм-боту.\n {traceback.format_exc()}")
        return

    if message.text.lower() in 'это не нормально это ужасно это очень плохо очень жаль кошмар охренеть как же так как жаль что случилось':
        bot.send_message(user_id, f"Ну, что поделать, {name}... Я тебя прекрасно понимаю.")
        bot.send_message(157758328, f"{fio} отправили сочувствие в ответ на {message.text}.")
        return

    if "сколько бортпроводников" in message.text.lower():
        bot.send_message(user_id, f"К Telegram-боту подключено сейчас {handler_db.count_users()} бортпроводников.")
        return

    if 'разослать сообщение' in message.text.lower():
        messaging_thread = threading.Thread(target=messaging_process(message))
        messaging_thread.start()
        return

    if "взаимно" in message.text.lower() or "и тебя" in message.text.lower():
        answer = "Спасибо большое!)"
        bot.send_message(user_id, answer)
        bot.send_message(157758328,
                         f"{fio} поблагодарил: {message.text} \n А бот ответил: {answer}", reply_markup=general_menu())
        return

    if "спасибо" in message.text.lower() or message.text.lower() in baza.good_bye:
        answer = choice(baza.best_wishes)
        bot.send_message(user_id, answer)
        bot.send_message(157758328,
                         f"{fio} поблагодарил: {message.text} \n А бот ответил: {answer}", reply_markup=general_menu())
        return

    if 'сохранить пользователей в excel' in message.text.lower() and message.chat.id == 157758328:
        handler_db.import_users_to_excel()
        bot.send_document(message.chat.id, open('general_db.xlsx', "rb"))
        return

    if 'исправить' in message.text:
        correct = f"Пользователь {fio} предложил правку: {message.text[10:]}"
        bot.send_message(user_id, 'Ваша информация успешно отправлена. После ее рассмотрения будут внесены '
                                  'соответствующие изменения. \n Большое спасибо за Ваше участие в улучшении '
                                  'Телеграм-Бота!', reply_markup=general_menu())
        bot.send_message(157758328, correct)
        return

    if message.text in "/addinfo добавить информацию":
        bot.send_message(user_id,
                         # TODO либо создавать новый словарь и методом в питон 3.9  а|b сливать его с существующим
                         'Для добавления своей информации в телеграм-бот, начните свое сообщение со слова "запомни:". '
                         'Например:\n\nЗапомни: номер телефона представителя в Москве 8(495)123-45-67',
                         reply_markup=general_menu())
        return

    if 'добавить' in message.text and message.text not in 'как добавиться в группу как добавить друга в группу':  # предложить заменили на добавить так как пересекается с предложить вино на английском языке
        correct = f"Пользователь {fio} предложил информацию: {message.text[9:]}"
        bot.send_message(user_id, f'{name}, Ваша информация успешно отправлена. После ее рассмотрения будут внесены '
                                  'соответсвующие изменения. \n Большое спасибо за Ваше участие в улучшении '
                                  'Телеграм-Бота!', reply_markup=general_menu())
        bot.send_message(157758328, correct)
        bot.send_message(1106606028, correct)
        return

    if 'инструктор' == message.text or 'инструктора' == message.text:
        bot.send_message(user_id, 'Какой именно инструктор Вас инетересует?', reply_markup=general_menu())
        return

    if 'телефон' == message.text or 'номер телефона' == message.text or "добавочный номер" == message.text or 'телефоны' in message.text or 'номера' in message.text:  # TODO наверное не очень семантично здесь размещать обработку этого запроса
        bot.send_message(user_id, 'Чей именно телефон Вас инетересует?', reply_markup=general_menu())
        return

    if 'почта' == message.text:
        bot.send_message(user_id, 'Чья именно почта Вас инетересует?', reply_markup=general_menu())
        return

    if 'особенности' == message.text:
        bot.send_message(user_id, 'Какие именно особенности Вас интересуют?', reply_markup=general_menu())
        return

    if 'особенности рейса' == message.text:
        bot.send_message(user_id, 'Какой город Вас интересует?', reply_markup=general_menu())
        return

    if 'питание' == message.text:
        bot.send_message(user_id, 'Какое питание Вас интересует?', reply_markup=general_menu())
        return

    if 'самолет' == message.text:
        bot.send_message(user_id, 'Какой самолет Вас интересует?', reply_markup=general_menu())
        return

    if 'супервайзер' == message.text:
        bot.send_message(user_id, 'Какой именно супервайзер Вас интересует?', reply_markup=general_menu())
        return

    ########## ai ПОШЕЛ САМ ТЕКСТ СЮДА СЛУШАЕТ ЗДЕСЬ

    try:
        logger.info(f"User {message.from_user.id} asked: {message.text}")

        db = handler_db.get_all_texts()

        if not db:
            bot.reply_to(message, "База знаний пуста. Сначала добавьте документы или обучающий текст.")
            return

        response = data_store.get_answer(message.text, message.from_user.id)
        logger.info(f"Generated response for user {message.from_user.id}")
        bot.reply_to(message, response) # , parse_mode="Markdown"
        if 'нет информации' in response:
            bot.send_message(157758328, f"{fio} не нашел ответ на вопрос \n\n{message.text}\n\n ему "
                                        f"предложили внести информацию")
            bot.reply_to(message, f'{name}, если вы знаете ответ на этот вопрос, пожалуйста, поделитесь своей '
                                  f'информацией. Начните свое сообщение со слова "Запомни", затем напишите свой вопрос и '
                                  f'подробный ответ (все это в рамках одного сообщения). \nСпасибо, что помогаете сделать '
                                  f'бота лучше и умнее.')
        # bot.send_message(157758328, f"ответ пользователю:\n   - {response}")
        # Сбрасываем состояние после обработки
        bot.delete_state(message.from_user.id, message.chat.id)

    except Exception as e:
        logger.error(f"Error generating response: {str(e)}\n traceback: {traceback.format_exc()}")
        bot.reply_to(s.ADMIN_ID, f'Произошла ошибка при генерации ответа:\n\n{traceback.format_exc()}')
        bot.reply_to(message, f"Произошла ошибка при генерации ответа {traceback}")


# Запуск бота
if __name__ == "__main__":
    logger.info("Bot started")
    try:
        logger.info("Starting polling...")
        bot.infinity_polling()
    except Exception as e:
        logger.error(f"Critical error: {str(e)}\n traceback: {traceback.format_exc()}")
