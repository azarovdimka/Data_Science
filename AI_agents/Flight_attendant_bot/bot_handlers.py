#!/usr/bin/env python3.9
"""
Обработчики команд и сообщений бота
"""
import logging
import traceback
from telebot import types
from random import choice
import PyPDF2
from data_store import DataStore
from pdf_processor import process_pdf_text
from config import KRS_LIST, ADMIN_ID
import settings as s

logger = logging.getLogger(__name__)


class BotHandlers:
    def __init__(self, bot):
        self.bot = bot
        self.data_store = DataStore()
        self.setup_handlers()
    
    def setup_handlers(self):
        """Настройка обработчиков команд и сообщений"""
        
        @self.bot.message_handler(commands=['start'])
        def handle_start(message):
            self.start_command(message)
        
        @self.bot.message_handler(content_types=['document'])
        def handle_document(message):
            self.document_handler(message)
        
        @self.bot.message_handler(func=lambda message: True)
        def handle_message(message):
            self.message_handler(message)
    
    def start_command(self, message):
        """Обработчик команды /start"""
        try:
            user_id = message.from_user.id
            username = message.from_user.username or "Пользователь"
            
            welcome_messages = [
                f"Привет, {username}! 👋\nЯ информационно-справочный бот для бортпроводников.",
                f"Добро пожаловать, {username}! ✈️\nЗадавайте вопросы по процедурам и регламентам.",
                f"Здравствуйте, {username}! 📋\nГотов помочь с информацией по работе бортпроводника."
            ]
            
            response = choice(welcome_messages)
            self.bot.reply_to(message, response)
            
            logger.info(f"Start command from user {user_id} ({username})")
            
        except Exception as e:
            logger.error(f"Error in start command: {str(e)}")
            self.bot.reply_to(message, "Произошла ошибка при запуске бота.")
    
    def document_handler(self, message):
        """Обработчик загруженных документов"""
        try:
            user_id = message.from_user.id
            
            # Проверка прав доступа
            if user_id not in KRS_LIST and user_id != ADMIN_ID:
                self.bot.reply_to(message, "У вас нет прав для загрузки документов.")
                return
            
            # Проверка типа файла
            if not message.document.file_name.lower().endswith('.pdf'):
                self.bot.reply_to(message, "Поддерживаются только PDF файлы.")
                return
            
            self.bot.reply_to(message, "Обрабатываю документ... ⏳")
            
            # Скачивание файла
            file_info = self.bot.get_file(message.document.file_id)
            downloaded_file = self.bot.download_file(file_info.file_path)
            
            # Обработка PDF
            try:
                # Сохранение временного файла
                temp_filename = f"temp_{message.document.file_name}"
                with open(temp_filename, 'wb') as temp_file:
                    temp_file.write(downloaded_file)
                
                # Извлечение текста из PDF
                with open(temp_filename, 'rb') as pdf_file:
                    pdf_reader = PyPDF2.PdfReader(pdf_file)
                    text = ""
                    for page in pdf_reader.pages:
                        text += page.extract_text()
                
                # Обработка и добавление в базу данных
                if process_pdf_text(text):
                    self.bot.reply_to(message, "✅ Документ успешно обработан и добавлен в базу знаний!")
                else:
                    self.bot.reply_to(message, "❌ Ошибка при обработке документа.")
                
                # Удаление временного файла
                import os
                os.remove(temp_filename)
                
            except Exception as pdf_error:
                logger.error(f"PDF processing error: {str(pdf_error)}")
                self.bot.reply_to(message, "Ошибка при обработке PDF файла.")
            
        except Exception as e:
            logger.error(f"Error in document handler: {str(e)}")
            self.bot.reply_to(message, "Произошла ошибка при обработке документа.")
    
    def message_handler(self, message):
        """Обработчик текстовых сообщений"""
        try:
            user_id = message.from_user.id
            username = message.from_user.username or "Unknown"
            question = message.text
            
            logger.info(f"Question from {username} ({user_id}): {question}")
            
            # Отправка индикатора "печатает"
            self.bot.send_chat_action(message.chat.id, 'typing')
            
            # Поиск и генерация ответа
            response = self.data_store.search_and_generate_response(question, user_id)
            
            # Отправка ответа
            self.bot.reply_to(message, response)
            
        except Exception as e:
            logger.error(f"Error in message handler: {str(e)}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            self.bot.reply_to(message, "Произошла ошибка при обработке вашего сообщения.")