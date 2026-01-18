"""
Скрипт для запуска Heart Attack Prediction API.
"""

import uvicorn
import os
import sys
from pathlib import Path

def main():
    """Основная функция запуска API."""
    
    # Добавление текущей директории в путь Python
    current_dir = Path(__file__).parent
    sys.path.insert(0, str(current_dir))
    
    # Создание необходимых директорий
    directories = ['models', 'data', 'data/predictions', 'static']
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
    
    print("🚀 Запуск Heart Attack Prediction API...")
    print("📍 API будет доступен по адресу: http://localhost:8001")
    print("📖 Документация API: http://localhost:8001/docs")
    print("🌐 Веб-интерфейс: http://localhost:8001")
    print("\n" + "="*50)
    
    # Запуск сервера
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8001,
        reload=True,
        log_level="info"
    )

if __name__ == "__main__":
    main()