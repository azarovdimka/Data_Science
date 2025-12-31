"""
Обработчик для работы с CSV файлами
"""
# TODO Мне кажется файл не используется - проверить!

from fastapi import File, UploadFile, HTTPException
from fastapi.responses import FileResponse
import pandas as pd
import io
import os
import chardet
from backend.services.ml_service import MLService  #  это экземпляр класса MLService из файла backend/services/ml_service.py.

ml_service = MLService()

async def upload_csv_file(file: UploadFile = File(...)):
    """Загрузка и обработка CSV файла"""
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="Файл должен быть в формате CSV")
    
    if file.size and file.size > 50 * 1024 * 1024:  # 50MB
        raise HTTPException(status_code=413, detail="Размер файла не должен превышать 50MB")
    
    try:
        contents = await file.read()
        # Автоматическое определение кодировки
        detected = chardet.detect(contents)
        encoding = detected['encoding'] or 'utf-8'
        
        # Попробуем разные кодировки
        encodings_to_try = [encoding, 'utf-8', 'cp1251', 'utf-8-sig', 'windows-1251']
        
        df = None
        for enc in encodings_to_try:
            try:
                df = pd.read_csv(io.StringIO(contents.decode(enc)), delimiter=';')
                break
            except (UnicodeDecodeError, UnicodeError):
                continue
        
        if df is None:
            raise HTTPException(status_code=400, detail="Не удалось определить кодировку файла")
        
        required_columns = [
            'Id экзамена', 'Id вопроса', '№ вопроса', 
            'Текст вопроса', 'Транскрибация ответа'
        ]
        
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise HTTPException(
                status_code=400, 
                detail=f"Отсутствуют обязательные колонки: {missing_columns}"
            )
        
        predictions = await ml_service.predict_scores(df)
        df['Оценка экзаменатора'] = predictions
        
        output_filename = f"results_{file.filename}"
        output_path = f"temp/{output_filename}"
        
        os.makedirs("temp", exist_ok=True)
        df.to_csv(output_path, index=False, sep=';', encoding='utf-8-sig')
        
        return {
            "message": "Файл успешно обработан",
            "filename": output_filename,
            "total_questions": len(df),
            "download_url": f"/download/{output_filename}"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка обработки файла: {str(e)}")

async def download_csv_file(filename: str):
    """Скачивание обработанного CSV файла"""
    file_path = f"temp/{filename}"
    
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Файл не найден")
    
    return FileResponse(
        path=file_path,
        filename=filename,
        media_type='text/csv'
    )