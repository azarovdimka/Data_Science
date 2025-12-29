"""
Основной файл FastAPI приложения для автоматической оценки экзаменов
"""

from fastapi import FastAPI, File, UploadFile, HTTPException, Request, WebSocket
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import pandas as pd
import io
import os
from typing import List, Dict
import uvicorn
import asyncio
import uuid

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.services.ml_service import MLService
from backend.services.audio_service import AudioService

try:
    from database.handlers.exam_handler import ExamHandler
    DB_AVAILABLE = True
except ImportError:
    DB_AVAILABLE = False
    print("База данных недоступна. Работа без БД.")
    
try:
    from config.settings import Settings
    settings = Settings()
except ImportError:
    class Settings:
        HOST = "0.0.0.0"
        PORT = 8000
        DEBUG = True
    settings = Settings()

# Инициализация приложения
app = FastAPI(
    title="Система автоматической оценки экзаменов",
    description="API для автоматической оценки устных ответов по русскому языку",
    version="1.0.0"
)

# Настройка middleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(TrustedHostMiddleware, allowed_hosts=["*"])
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    max_age=600,
)

# Подключение статических файлов и шаблонов
app.mount("/static", StaticFiles(directory="frontend/static"), name="static")
templates = Jinja2Templates(directory="frontend/templates")

# Инициализация сервисов
ml_service = MLService()
audio_service = AudioService()

if DB_AVAILABLE:
    exam_handler = ExamHandler()
else:
    exam_handler = None

# Хранилище задач и WebSocket соединений
processing_tasks: Dict[str, dict] = {}
active_connections: Dict[str, WebSocket] = {}

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """Главная страница"""
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/upload-csv/")
async def upload_csv(file: UploadFile = File(...)):
    """
    Загрузка CSV файла для обработки (асинхронно)
    """
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="Файл должен быть в формате CSV")
    
    if file.size and file.size > 50 * 1024 * 1024:  # 50MB
        raise HTTPException(status_code=413, detail="Размер файла не должен превышать 50MB")
    
    # Создаем уникальный ID задачи
    task_id = str(uuid.uuid4())
    
    try:
        # Чтение CSV файла
        contents = await file.read()
        
        # Автоматическое определение кодировки
        import chardet
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
        
        # # Валидация структуры файла
        # required_columns = [
        #     'Id экзамена', 'Id вопроса', '№ вопроса', 
        #     'Текст вопроса', 'Транскрибация ответа'
        # ]
        
        # missing_columns = [col for col in required_columns if col not in df.columns]
        # if missing_columns:
        #     raise HTTPException(
        #         status_code=400, 
        #         detail=f"Отсутствуют обязательные колонки: {missing_columns}"
        #     )
        
        # Сохраняем информацию о задаче
        processing_tasks[task_id] = {
            "status": "processing",
            "progress": 0,
            "message": "Начинаем обработку...",
            "filename": file.filename,
            "total_questions": len(df)
        }
        
        # Запускаем обработку в фоне
        asyncio.create_task(process_csv_background(task_id, df, file.filename))
        
        return {
            "task_id": task_id,
            "message": "Обработка запущена",
            "status_url": f"/status/{task_id}"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка обработки файла: {str(e)}")

async def process_csv_background(task_id: str, df: pd.DataFrame, filename: str):
    """Фоновая обработка CSV файла"""
    try:
        # Обновляем прогресс
        processing_tasks[task_id]["progress"] = 10
        processing_tasks[task_id]["message"] = "Обработка данных через ML модель..."
        await broadcast_progress(task_id, 10, "Обработка данных через ML модель...")
        
        # Обработка данных через ML модель с передачей прогресса
        async def progress_callback(percent, message):
            processing_tasks[task_id]["progress"] = percent
            processing_tasks[task_id]["message"] = message
            await broadcast_progress(task_id, percent, message)
        
        predictions = await ml_service.predict_scores(df, progress_callback)
        
        # Обновляем прогресс
        processing_tasks[task_id]["progress"] = 90
        processing_tasks[task_id]["message"] = "Сохранение результатов..."
        await broadcast_progress(task_id, 90, "Сохранение результатов...")
        
        # Добавление предсказаний в DataFrame
        df['Оценка экзаменатора'] = predictions
        
        # Сохранение результата
        output_filename = f"results_{filename}"
        temp_dir = os.path.abspath("temp")
        output_path = os.path.join(temp_dir, output_filename)
        
        os.makedirs(temp_dir, exist_ok=True)
        df.to_csv(output_path, index=False, sep=';', encoding='utf-8-sig')
        
        # Обновляем статус задачи
        processing_tasks[task_id] = {
            "status": "completed",
            "progress": 100,
            "message": "Обработка завершена",
            "filename": output_filename,
            "download_url": f"/download/{output_filename}",
            "total_questions": len(df)
        }
        await broadcast_progress(task_id, 100, "Обработка завершена")
        
    except Exception as e:
        processing_tasks[task_id] = {
            "status": "error",
            "progress": 0,
            "message": f"Ошибка: {str(e)}"
        }

@app.get("/status/{task_id}")
async def get_task_status(task_id: str):
    """Получение статуса обработки задачи"""
    if task_id not in processing_tasks:
        raise HTTPException(status_code=404, detail="Задача не найдена")
    
    return processing_tasks[task_id]

@app.websocket("/ws/progress")
async def websocket_progress(websocket: WebSocket):
    """Трансляция прогресса через WebSocket"""
    await websocket.accept()
    connection_id = str(uuid.uuid4())
    active_connections[connection_id] = websocket
    
    try:
        while True:
            # Ожидаем сообщения от клиента
            await websocket.receive_text()
    except:
        # Удаляем соединение при отключении
        if connection_id in active_connections:
            del active_connections[connection_id]

async def broadcast_progress(task_id: str, progress: int, message: str):
    """Отправка прогресса всем подключенным клиентам"""
    if not active_connections:
        return
        
    message_data = {
        "type": "progress",
        "task_id": task_id,
        "progress": progress,
        "message": message
    }
    
    # Отправляем всем подключенным клиентам
    disconnected = []
    for connection_id, websocket in active_connections.items():
        try:
            await websocket.send_json(message_data)
        except:
            disconnected.append(connection_id)
    
    # Удаляем отключенные соединения
    for connection_id in disconnected:
        del active_connections[connection_id]

@app.get("/download/{filename}")
async def download_file(filename: str):
    """
    Скачивание обработанного файла
    """
    file_path = os.path.abspath(f"temp/{filename}")
    
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Файл не найден")
    
    return FileResponse(
        path=file_path,
        filename=filename,
        media_type='text/csv',
        headers={"Content-Disposition": f"attachment; filename={filename}"}
    )

@app.post("/predict-single/")
async def predict_single(
    question_number: int,
    question_text: str,
    transcription: str
):
    """
    Предсказание оценки для одного вопроса
    """
    try:
        score = await ml_service.predict_single(
            question_number=question_number,
            question_text=question_text,
            transcription=transcription
        )
        
        return {
            "question_number": question_number,
            "predicted_score": score,
            "max_score": ml_service.get_max_score(question_number)
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка предсказания: {str(e)}")

# @app.post("/process-audio/")
# async def process_audio(file: UploadFile = File(...)):
#     """
#     Обработка аудиофайла и получение транскрипции
#     """
#     if not file.content_type.startswith('audio/'):
#         raise HTTPException(status_code=400, detail="Файл должен быть аудиоформата")
    
#     try:
#         # Сохранение временного файла
#         temp_path = f"temp/audio_{file.filename}"
#         os.makedirs("temp", exist_ok=True)
        
#         with open(temp_path, "wb") as buffer:
#             content = await file.read()
#             buffer.write(content)
        
#         # Получение транскрипции
#         transcription = await audio_service.transcribe_audio(temp_path)
        
#         # Удаление временного файла
#         os.remove(temp_path)
        
#         return {
#             "transcription": transcription,
#             "filename": file.filename
#         }
        
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=f"Ошибка обработки аудио: {str(e)}")

# @app.get("/api/stats")
# async def get_statistics():
#     """
#     Получение статистики системы
#     """
#     if not DB_AVAILABLE or not exam_handler:
#         return {"message": "База данных недоступна"}
    
#     try:
#         stats = await exam_handler.get_statistics()
#         return stats
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=f"Ошибка получения статистики: {str(e)}")

@app.get("/health")
async def health_check():
    """
    Проверка состояния системы
    """
    db_connected = False
    if DB_AVAILABLE and exam_handler:
        try:
            db_connected = await exam_handler.check_connection()
        except:
            db_connected = False
    
    return {
        "status": "healthy",
        "ml_model_loaded": ml_service.is_model_loaded(),
        "database_connected": db_connected
    }

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=False
    )