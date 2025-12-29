#!/usr/bin/env python3.9
"""
Модуль для обработки PDF файлов
"""
import re
import logging
import handler_db

logger = logging.getLogger(__name__)


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