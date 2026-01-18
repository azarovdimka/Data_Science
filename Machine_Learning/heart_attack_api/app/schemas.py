"""
Схемы данных для API предсказания сердечных приступов.
"""

from pydantic import BaseModel, Field
from typing import Dict, Optional, List, Any

class HealthCheck(BaseModel):
    """Схема для проверки состояния API"""
    status: str = Field(..., description="Статус API")
    message: str = Field(..., description="Сообщение о состоянии")

class PredictionResponse(BaseModel):
    """Схема ответа с предсказаниями"""
    predictions: Dict[str, int] = Field(..., description="Словарь предсказаний {id: prediction}")
    total_predictions: int = Field(..., description="Общее количество предсказаний")
    message: str = Field(..., description="Сообщение о результате")

class PatientData(BaseModel):
    """Схема данных пациента"""
    Age: float = Field(..., description="Возраст (нормализованный)")
    Cholesterol: float = Field(..., description="Уровень холестерина (нормализованный)")
    Heart_rate: float = Field(..., description="Частота сердечных сокращений (нормализованная)", alias="Heart rate")
    Diabetes: float = Field(..., description="Диабет (0 или 1)")
    Family_History: float = Field(..., description="Семейный анамнез (0 или 1)", alias="Family History")
    Smoking: float = Field(..., description="Курение (0 или 1)")
    Obesity: float = Field(..., description="Ожирение (0 или 1)")
    Alcohol_Consumption: float = Field(..., description="Употребление алкоголя (0 или 1)", alias="Alcohol Consumption")
    Exercise_Hours_Per_Week: float = Field(..., description="Часы упражнений в неделю (нормализованные)", alias="Exercise Hours Per Week")
    Diet: int = Field(..., description="Тип диеты (0-3)")
    Previous_Heart_Problems: float = Field(..., description="Предыдущие проблемы с сердцем (0 или 1)", alias="Previous Heart Problems")
    Medication_Use: float = Field(..., description="Использование лекарств (0 или 1)", alias="Medication Use")
    Stress_Level: float = Field(..., description="Уровень стресса (1-10)", alias="Stress Level")
    Sedentary_Hours_Per_Day: float = Field(..., description="Малоподвижные часы в день (нормализованные)", alias="Sedentary Hours Per Day")
    Income: float = Field(..., description="Доход (нормализованный)")
    BMI: float = Field(..., description="ИМТ (нормализованный)")
    Triglycerides: float = Field(..., description="Триглицериды (нормализованные)")
    Physical_Activity_Days_Per_Week: float = Field(..., description="Дни физической активности в неделю", alias="Physical Activity Days Per Week")
    Sleep_Hours_Per_Day: float = Field(..., description="Часы сна в день (нормализованные)", alias="Sleep Hours Per Day")
    Blood_sugar: float = Field(..., description="Уровень сахара в крови (нормализованный)", alias="Blood sugar")
    CK_MB: float = Field(..., description="КК-МБ (нормализованный)", alias="CK-MB")
    Troponin: float = Field(..., description="Тропонин (нормализованный)")
    Gender: str = Field(..., description="Пол (Male/Female)")
    Systolic_blood_pressure: float = Field(..., description="Систолическое давление (нормализованное)", alias="Systolic blood pressure")
    Diastolic_blood_pressure: float = Field(..., description="Диастолическое давление (нормализованное)", alias="Diastolic blood pressure")

class PredictionRequest(BaseModel):
    """Схема запроса на предсказание"""
    patient_data: PatientData = Field(..., description="Данные пациента")

class BatchPredictionResponse(BaseModel):
    """Схема ответа для пакетных предсказаний"""
    predictions: List[Dict[str, Any]] = Field(..., description="Список предсказаний")
    summary: Dict[str, Any] = Field(..., description="Сводная информация")
    
class ErrorResponse(BaseModel):
    """Схема ответа об ошибке"""
    error: str = Field(..., description="Тип ошибки")
    message: str = Field(..., description="Сообщение об ошибке")
    details: Optional[str] = Field(None, description="Дополнительные детали ошибки")