"""
FastAPI приложение для предсказания риска сердечного приступа.
"""

from fastapi import FastAPI, HTTPException, UploadFile, File, Request
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
import os
import tempfile
from pathlib import Path
import logging

from .models.predictor import HeartAttackPredictor
from .models.data_processor import DataProcessor
from .schemas import PredictionResponse, HealthCheck

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Heart Attack Risk Prediction API",
    description="API для предсказания риска сердечного приступа на основе медицинских показателей",
    version="1.0.0"
)

# Инициализация компонентов
try:
    predictor: HeartAttackPredictor = HeartAttackPredictor()
    logger.info("Predictor инициализирован успешно")
except Exception as e:
    logger.error(f"Ошибка инициализации Predictor: {e}")
    predictor = None

# Настройка статических файлов и шаблонов
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

@app.get("/", response_class=FileResponse)
async def read_root() -> FileResponse:
    """Главная страница с веб-интерфейсом.
    
    Returns:
        FileResponse: HTML страница с веб-интерфейсом
    """
    return FileResponse("templates/index.html")

@app.get("/health", response_model=HealthCheck)
async def health_check() -> HealthCheck:
    """Проверка состояния API.
    
    Returns:
        HealthCheck: Статус работы API
    """
    logger.info("Health check запрос")
    
    status = "healthy" if predictor and predictor.is_loaded else "unhealthy"
    message = "API работает корректно" if status == "healthy" else "Модель не загружена"
    
    return HealthCheck(status=status, message=message)

@app.post("/predict/csv", response_model=PredictionResponse)
async def predict_from_csv(file: UploadFile = File(...)) -> PredictionResponse:
    """
    Предсказание риска сердечного приступа из CSV файла.
    Применяет ту же предобработку что и в тетрадке.
    
    Args:
        file (UploadFile): CSV файл с данными пациентов
        
    Returns:
        PredictionResponse: JSON с предсказаниями в формате {id: prediction}
        
    Raises:
        HTTPException: При ошибках валидации или обработки файла
    """
    if not predictor or not predictor.is_loaded:
        raise HTTPException(status_code=503, detail="Модель не загружена")
    
    logger.info(f"Получен CSV файл: {file.filename}")
    
    try:
        # Проверка типа файла
        if not file.filename or not file.filename.endswith('.csv'):
            raise HTTPException(status_code=400, detail="Файл должен быть в формате CSV")
        
        # Чтение файла
        contents: bytes = await file.read()
        
        # Создание временного файла
        with tempfile.NamedTemporaryFile(mode='wb', delete=False, suffix='.csv') as tmp_file:
            tmp_file.write(contents)
            tmp_path: str = tmp_file.name
        
        try:
            # Загрузка и обработка данных
            df: pd.DataFrame = pd.read_csv(tmp_path)
            logger.info(f"Загружено {len(df)} записей из CSV")
            
            # Проверка наличия необходимых колонок
            required_cols = ['Age', 'BMI', 'Diabetes', 'Systolic blood pressure', 'Diastolic blood pressure']
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                raise HTTPException(
                    status_code=400, 
                    detail=f"Отсутствуют обязательные колонки: {missing_cols}"
                )
            
            # Пакетное предсказание
            if 'id' not in df.columns:
                df['id'] = range(len(df))
            
            results = predictor.predict_batch(df)
            
            # Формирование результата в нужном формате
            predictions_dict = {str(result['id']): result['prediction'] for result in results}
            
            logger.info(f"Выполнено {len(results)} предсказаний")
            
            return PredictionResponse(
                predictions=predictions_dict,
                total_predictions=len(results),
                message="Предсказания успешно выполнены с полной предобработкой как в тетрадке"
            )
            
        finally:
            # Удаление временного файла
            os.unlink(tmp_path)
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Ошибка при обработке CSV файла: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Ошибка при обработке файла: {str(e)}")

@app.post("/predict/json")
async def predict_from_json(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Предсказание риска сердечного приступа из JSON данных.
    Применяет ту же предобработку что и в тетрадке.
    
    Args:
        data (Dict[str, Any]): JSON с данными пациента
        
    Returns:
        Dict[str, Any]: JSON с предсказанием и уровнем риска
        
    Raises:
        HTTPException: При ошибках обработки данных
    """
    if not predictor or not predictor.is_loaded:
        raise HTTPException(status_code=503, detail="Модель не загружена")
    
    logger.info("Получен JSON запрос на предсказание")
    
    try:
        # Предсказание для одного пациента
        result = predictor.predict_single(data)
        
        logger.info(f"Предсказание выполнено: {result['prediction']}")
        
        return {
            "prediction": result['prediction'],
            "probability": result['probability'],
            "risk_level": "Высокий риск" if result['prediction'] == 1 else "Низкий риск",
            "message": "Предсказание выполнено с полной предобработкой как в тетрадке"
        }
        
    except ValueError as e:
        logger.error(f"Ошибка валидации данных: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Ошибка при обработке JSON данных: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Ошибка при обработке данных: {str(e)}")

@app.get("/download/predictions/{filename}")
async def download_predictions(filename: str) -> FileResponse:
    """
    Скачивание файла с предсказаниями.
    
    Args:
        filename (str): Имя файла для скачивания
        
    Returns:
        FileResponse: Файл с предсказаниями
        
    Raises:
        HTTPException: Если файл не найден
    """
    file_path: str = f"data/predictions/{filename}"
    
    if os.path.exists(file_path):
        logger.info(f"Скачивание файла: {filename}")
        return FileResponse(file_path, filename=filename)
    else:
        logger.warning(f"Файл не найден: {filename}")
        raise HTTPException(status_code=404, detail="Файл не найден")

@app.get("/model/info")
async def get_model_info() -> Dict[str, Any]:
    """
    Информация о модели и предобработке.
    
    Returns:
        Dict[str, Any]: Информация о модели, предобработке и версии
    """
    logger.info("Запрос информации о модели")
    
    if predictor:
        model_info = predictor.get_model_info()
        model_info.update({
            "preprocessing": "Полная предобработка как в тетрадке с feature engineering",
            "features_created": "9 отобранных признаков (bmi_diabetes, pressure_product, и др.)",
            "feature_selection": "Корреляционный отбор (топ-9 признаков)",
            "scaling": "StandardScaler",
            "medical_optimization": f"Оптимальный порог {model_info.get('optimal_threshold', 0.5)} для высокого recall",
            "recall_target": "92%+ для медицинской диагностики",
            "version": "1.0.0",
            "description": "Модель для предсказания риска сердечного приступа с полной предобработкой данных"
        })
        return model_info
    else:
        return {
            "model_type": "CatBoost Classifier",
            "is_loaded": False,
            "error": "Модель не загружена",
            "version": "1.0.0"
        }

@app.get("/predict/sample")
async def get_sample_prediction() -> Dict[str, Any]:
    """
    Пример предсказания с тестовыми данными.
    
    Returns:
        Dict[str, Any]: Результат предсказания на тестовых данных
    """
    if not predictor or not predictor.is_loaded:
        raise HTTPException(status_code=503, detail="Модель не загружена")
    
    # Получаем пример данных из процессора
    processor = DataProcessor()
    sample_data = processor.get_sample_data()
    
    try:
        result = predictor.predict_single(sample_data)
        
        return {
            "sample_data": sample_data,
            "prediction": result['prediction'],
            "probability": result['probability'],
            "risk_level": "Высокий риск" if result['prediction'] == 1 else "Низкий риск",
            "message": "Пример предсказания на тестовых данных"
        }
        
    except Exception as e:
        logger.error(f"Ошибка при тестовом предсказании: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Ошибка при тестовом предсказании: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)