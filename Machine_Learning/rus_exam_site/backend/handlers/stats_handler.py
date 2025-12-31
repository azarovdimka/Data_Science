"""
Обработчик для статистики и аналитики
"""
# TODO Мне кажется файл не используется - проверить

from fastapi import HTTPException

try:
    from database.handlers.exam_handler import ExamHandler
    DB_AVAILABLE = True
    exam_handler = ExamHandler()
except ImportError:
    DB_AVAILABLE = False
    exam_handler = None

async def get_system_statistics():
    """Получение статистики системы"""
    if not DB_AVAILABLE or not exam_handler:
        return {"message": "База данных недоступна"}
    
    try:
        stats = await exam_handler.get_statistics()
        return stats
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка получения статистики: {str(e)}")

async def get_health_status():
    """Проверка состояния системы"""
    from backend.services.ml_service import MLService
    
    ml_service = MLService()
    
    db_connected = False
    if DB_AVAILABLE and exam_handler:
        try:
            db_connected = await exam_handler.check_connection()
        except:
            db_connected = False
    
    return {
        "status": "healthy",
        "ml_model_loaded": ml_service.is_model_loaded(),
        "database_connected": db_connected,
        "components": {
            "ml_service": "active" if ml_service.is_model_loaded() else "inactive",
            "database": "connected" if db_connected else "disconnected"
        }
    }