"""
Конфигурационные настройки приложения
"""

import os
from typing import Optional
from pydantic import BaseSettings

class Settings(BaseSettings):
    """Настройки приложения"""
    
    # Основные настройки
    APP_NAME: str = "Система автоматической оценки экзаменов"
    VERSION: str = "1.0.0"
    DEBUG: bool = True
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    
    # База данных
    DATABASE_URL: str = "postgresql://user:password@localhost:5432/exam_db"
    DB_HOST: str = "localhost"
    DB_PORT: int = 5432
    DB_NAME: str = "exam_db"
    DB_USER: str = "exam_user"
    DB_PASSWORD: str = "exam_password"
    
    # Машинное обучение
    ML_MODEL_PATH: str = "ml_models/models/exam_scorer.pkl"
    VECTORIZER_PATH: str = "ml_models/models/vectorizer.pkl"
    USE_PRETRAINED_MODEL: bool = True
    
    # Аудио обработка
    AUDIO_UPLOAD_DIR: str = "uploads/audio"
    MAX_AUDIO_SIZE: int = 50 * 1024 * 1024  # 50MB
    SUPPORTED_AUDIO_FORMATS: list = [".mp3", ".wav", ".m4a", ".ogg"]
    
    # Файлы
    UPLOAD_DIR: str = "uploads"
    TEMP_DIR: str = "temp"
    MAX_FILE_SIZE: int = 50 * 1024 * 1024  # 50MB
    
    # Безопасность
    SECRET_KEY: str = "your-secret-key-here"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    
    # Внешние API
    SPEECH_RECOGNITION_API_KEY: Optional[str] = None
    YANDEX_SPEECH_API_KEY: Optional[str] = None
    
    # Логирование
    LOG_LEVEL: str = "INFO"
    LOG_FILE: str = "logs/app.log"
    
    class Config:
        env_file = ".env"
        case_sensitive = True

# Создание экземпляра настроек
settings = Settings()

# Настройки для разных окружений
class DevelopmentSettings(Settings):
    DEBUG: bool = True
    LOG_LEVEL: str = "DEBUG"

class ProductionSettings(Settings):
    DEBUG: bool = False
    LOG_LEVEL: str = "WARNING"
    HOST: str = "0.0.0.0"

class TestingSettings(Settings):
    DEBUG: bool = True
    DATABASE_URL: str = "postgresql://test_user:test_password@localhost:5432/test_exam_db"
    
def get_settings() -> Settings:
    """Получение настроек в зависимости от окружения"""
    env = os.getenv("ENVIRONMENT", "development").lower()
    
    if env == "production":
        return ProductionSettings()
    elif env == "testing":
        return TestingSettings()
    else:
        return DevelopmentSettings()