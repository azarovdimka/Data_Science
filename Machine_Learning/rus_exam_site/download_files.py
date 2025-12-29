#!/usr/bin/env python3
"""
Скрипт для скачивания файлов без зависимостей от requests
"""

import urllib.request
import os

def download_file(url, filename):
    """Скачивание файла через urllib"""
    try:
        print(f"Скачиваем {filename}...")
        urllib.request.urlretrieve(url, filename)
        print(f"✓ {filename} скачан")
        return True
    except Exception as e:
        print(f"✗ Ошибка скачивания {filename}: {e}")
        return False

def main():
    # Создаем папку models если её нет
    os.makedirs("backend/models", exist_ok=True)
    
    # Список файлов для скачивания
    files = [
        {
            "url": "https://example.com/evaluate_exam_model.pkl",
            "filename": "backend/models/evaluate_exam_model.pkl"
        }
    ]
    
    for file_info in files:
        download_file(file_info["url"], file_info["filename"])

if __name__ == "__main__":
    main()