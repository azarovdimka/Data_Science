"""
FastAPI приложение для предсказания риска сердечного приступа.
"""

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi import Request
import pandas as pd
import numpy as np
from typing import Dict, List, Any
import os
import tempfile
from pathlib import Path

from .models.predictor import HeartAttackPredictor
from .models.data_processor import DataProcessor
from .schemas import PredictionResponse, HealthCheck

app = FastAPI(
    title="Heart Attack Risk Prediction API",
    description="API для предсказания риска сердечного приступа на основе медицинских показателей",
    version="1.0.0"
)

# Инициализация компонентов
predictor = HeartAttackPredictor()
data_processor = DataProcessor()

# Настройка статических файлов и шаблонов
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

@app.get("/", response_class=FileResponse)
async def read_root():
    """Главная страница с веб-интерфейсом"""
    return FileResponse("templates/index.html")

@app.get("/health", response_model=HealthCheck)
async def health_check():
    """Проверка состояния API"""
    return HealthCheck(status="healthy", message="API работает корректно")

@app.post("/predict/csv", response_model=PredictionResponse)
async def predict_from_csv(file: UploadFile = File(...)):
    """
    Предсказание риска сердечного приступа из CSV файла.
    
    Args:
        file: CSV файл с данными пациентов
        
    Returns:
        JSON с предсказаниями в формате {id: prediction}
    """
    try:
        # Проверка типа файла
        if not file.filename.endswith('.csv'):
            raise HTTPException(status_code=400, detail="Файл должен быть в формате CSV")
        
        # Чтение файла
        contents = await file.read()
        
        # Создание временного файла
        with tempfile.NamedTemporaryFile(mode='wb', delete=False, suffix='.csv') as tmp_file:
            tmp_file.write(contents)
            tmp_path = tmp_file.name
        
        try:
            # Загрузка и обработка данных
            df = pd.read_csv(tmp_path)
            
            # Предобработка данных
            processed_df = data_processor.preprocess(df)
            
            # Получение предсказаний
            predictions = predictor.predict(processed_df)
            
            # Формирование результата
            if 'id' in df.columns:
                result = {
                    str(row['id']): int(pred) 
                    for _, row in df.iterrows() 
                    for pred in [predictions[_]]
                }
            else:
                result = {str(i): int(pred) for i, pred in enumerate(predictions)}
            
            return PredictionResponse(
                predictions=result,
                total_predictions=len(predictions),
                message="Предсказания успешно выполнены"
            )
            
        finally:
            # Удаление временного файла
            os.unlink(tmp_path)
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка при обработке файла: {str(e)}")

@app.post("/predict/json")
async def predict_from_json(data: Dict[str, Any]):
    """
    Предсказание риска сердечного приступа из JSON данных.
    
    Args:
        data: JSON с данными пациента
        
    Returns:
        JSON с предсказанием
    """
    try:
        # Преобразование в DataFrame
        df = pd.DataFrame([data])
        
        # Предобработка данных
        processed_df = data_processor.preprocess(df)
        
        # Получение предсказания
        prediction = predictor.predict(processed_df)[0]
        
        return {
            "prediction": int(prediction),
            "risk_level": "Высокий риск" if prediction == 1 else "Низкий риск",
            "message": "Предсказание успешно выполнено"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка при обработке данных: {str(e)}")

@app.get("/download/predictions/{filename}")
async def download_predictions(filename: str):
    """Скачивание файла с предсказаниями"""
    file_path = f"data/predictions/{filename}"
    if os.path.exists(file_path):
        return FileResponse(file_path, filename=filename)
    else:
        raise HTTPException(status_code=404, detail="Файл не найден")

@app.get("/model/info")
async def get_model_info():
    """Информация о модели"""
    return {
        "model_type": "CatBoost Classifier",
        "features": data_processor.get_feature_names(),
        "version": "1.0.0",
        "description": "Модель для предсказания риска сердечного приступа"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)