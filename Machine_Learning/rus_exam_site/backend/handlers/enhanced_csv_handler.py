"""
Улучшенный обработчик CSV файлов с полной предобработкой
"""

from fastapi import File, UploadFile, HTTPException
from fastapi.responses import FileResponse
import pandas as pd
import io
import os
from backend.services.enhanced_ml_service import EnhancedMLService

# Создаем экземпляр улучшенного ML сервиса
enhanced_ml_service = EnhancedMLService()

async def upload_csv_file_enhanced(file: UploadFile = File(...)):
    """Загрузка и обработка CSV файла с полной предобработкой"""
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="Файл должен быть в формате CSV")
    
    if file.size and file.size > 100 * 1024 * 1024:  # 100MB для аудиофайлов
        raise HTTPException(status_code=413, detail="Размер файла не должен превышать 100MB")
    
    try:
        contents = await file.read()
        
        # Попробуем разные кодировки
        for encoding in ['utf-8', 'cp1251', 'latin1']:
            try:
                df = pd.read_csv(io.StringIO(contents.decode(encoding)), delimiter=';')
                break
            except UnicodeDecodeError:
                continue
        else:
            raise HTTPException(status_code=400, detail="Не удалось определить кодировку файла")
        
        # Проверяем обязательные колонки
        required_columns = [
            '№ вопроса', 
            'Текст вопроса', 
            'Транскрибация ответа'
        ]
        
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise HTTPException(
                status_code=400, 
                detail=f"Отсутствуют обязательные колонки: {missing_columns}"
            )
        
        # Проверяем, что нет целевой переменной (как и должно быть)
        if 'Оценка экзаменатора' in df.columns:
            raise HTTPException(
                status_code=400, 
                detail="Файл не должен содержать колонку 'Оценка экзаменатора'. Загрузите файл без целевой переменной."
            )
        
        # Добавляем недостающие колонки если их нет
        if 'Id экзамена' not in df.columns:
            df['Id экзамена'] = range(1, len(df) + 1)
        
        if 'Id вопроса' not in df.columns:
            df['Id вопроса'] = range(1, len(df) + 1)
        
        print(f"Обрабатываем файл с {len(df)} записями...")
        
        # Получаем предсказания с полной предобработкой
        predictions = await enhanced_ml_service.predict_scores(df)
        
        # Добавляем предсказания к исходному DataFrame
        result_df = df.copy()
        result_df['Оценка экзаменатора'] = predictions
        
        # Сохраняем результат
        output_filename = f"results_{file.filename}"
        output_path = f"temp/{output_filename}"
        
        os.makedirs("temp", exist_ok=True)
        result_df.to_csv(output_path, index=False, sep=';', encoding='utf-8')
        
        # Статистика предсказаний
        prediction_stats = {
            'total_questions': len(result_df),
            'score_distribution': result_df['Оценка экзаменатора'].value_counts().to_dict(),
            'average_score': float(result_df['Оценка экзаменатора'].mean()),
            'questions_by_number': result_df['№ вопроса'].value_counts().to_dict()
        }
        
        return {
            "message": "Файл успешно обработан с полной предобработкой данных",
            "filename": output_filename,
            "statistics": prediction_stats,
            "download_url": f"/download/{output_filename}",
            "processing_info": {
                "audio_processed": 'Ссылка на оригинальный файл запис' in df.columns,
                "features_created": "Созданы признаки качества речи, морфологический анализ, взаимодействия",
                "model_type": "Enhanced RandomForest with full preprocessing pipeline"
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка обработки файла: {str(e)}")

async def download_csv_file_enhanced(filename: str):
    """Скачивание обработанного CSV файла"""
    file_path = f"temp/{filename}"
    
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Файл не найден")
    
    return FileResponse(
        path=file_path,
        filename=filename,
        media_type='text/csv',
        headers={"Content-Disposition": f"attachment; filename={filename}"}
    )

async def get_processing_status():
    """Получение статуса обработки и информации о модели"""
    return {
        "model_loaded": enhanced_ml_service.is_model_loaded(),
        "model_type": "Enhanced ML Service with full preprocessing",
        "supported_features": [
            "Аудио обработка (длительность файлов)",
            "Морфологический анализ текста",
            "Качество речи и лексическое разнообразие", 
            "Признаки взаимодействия",
            "TF-IDF векторизация",
            "Автоматическое масштабирование признаков"
        ],
        "question_max_scores": enhanced_ml_service.question_max_scores
    }

async def get_model_info():
    """Получение информации о модели"""
    try:
        feature_importance = enhanced_ml_service.get_feature_importance()
        
        return {
            "model_loaded": enhanced_ml_service.is_model_loaded(),
            "feature_importance": feature_importance,
            "preprocessing_steps": [
                "AudioProcessor - обработка аудиофайлов",
                "DataPreprocessor - создание признаков качества речи",
                "ColumnTransformer - обработка числовых, текстовых и категориальных признаков",
                "SelectKBest - отбор лучших признаков",
                "RobustScaler - масштабирование",
                "RandomForestClassifier - классификация"
            ]
        }
    except Exception as e:
        return {
            "error": f"Ошибка получения информации о модели: {str(e)}",
            "model_loaded": False
        }