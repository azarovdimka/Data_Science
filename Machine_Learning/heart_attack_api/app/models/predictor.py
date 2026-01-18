"""
Класс для предсказания риска сердечного приступа.
Использует обученную модель CatBoost из тетрадки.
"""

import pandas as pd
import numpy as np
from catboost import CatBoostClassifier
import joblib
import os
from typing import List, Optional, Dict, Any
import logging

class HeartAttackPredictor:
    """
    Класс для предсказания риска сердечного приступа с использованием 
    обученной модели CatBoost из тетрадки.
    """
    
    def __init__(self, model_path: Optional[str] = None) -> None:
        """
        Инициализация предиктора.
        
        Args:
            model_path: Путь к сохраненной модели
        """
        self.model: Optional[CatBoostClassifier] = None
        self.is_loaded: bool = False
        self.model_path: str = model_path or "models/heart_attack_model.cbm"
        
        # Настройка логирования
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Автоматическая загрузка модели при инициализации
        self.load_model()
    
    def load_model(self, path: Optional[str] = None) -> None:
        """
        Загрузка обученной модели из тетрадки.
        
        Args:
            path: Путь к модели
            
        Raises:
            FileNotFoundError: Если файл модели не найден
        """
        load_path = path or self.model_path
        
        try:
            if not os.path.exists(load_path):
                self.logger.warning(f"Файл модели не найден: {load_path}")
                self.logger.info("Создаю модель-заглушку для демонстрации")
                self._create_dummy_model()
                return
            
            # Загрузка модели CatBoost
            self.model = CatBoostClassifier()
            self.model.load_model(load_path)
            self.is_loaded = True
            
            self.logger.info(f"Модель успешно загружена: {load_path}")
            
        except Exception as e:
            self.logger.error(f"Ошибка при загрузке модели: {str(e)}")
            self.logger.info("Создаю модель-заглушку для демонстрации")
            self._create_dummy_model()
    
    def _create_dummy_model(self) -> None:
        """Создание модели-заглушки для демонстрации работы API."""
        self.model = CatBoostClassifier(
            iterations=100,
            learning_rate=0.1,
            depth=4,
            random_seed=42,
            verbose=False
        )
        
        # Создаем фиктивные данные для "обучения" заглушки
        np.random.seed(42)
        X_dummy = pd.DataFrame(np.random.rand(100, 9), columns=[
            'bmi_diabetes', 'pressure_product', 'lifestyle_score', 'trig_sedentary',
            'exercise_age1', 'bmi_exercise1', 'Systolic blood pressure', 
            'cardiac_markers2', 'chol_income'
        ])
        y_dummy = np.random.randint(0, 2, 100)
        
        self.model.fit(X_dummy, y_dummy, verbose=False)
        self.is_loaded = True
        
        self.logger.info("Создана модель-заглушка для демонстрации")
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Предсказание риска сердечного приступа.
        
        Args:
            X: Обработанные данные для предсказания (после DataProcessor)
            
        Returns:
            np.ndarray: Массив предсказаний (0 - низкий риск, 1 - высокий риск)
            
        Raises:
            ValueError: Если модель не загружена
        """
        if not self.is_loaded or self.model is None:
            raise ValueError("Модель не загружена. Проверьте путь к файлу модели.")
        
        try:
            predictions = self.model.predict(X)
            return predictions.astype(int)
            
        except Exception as e:
            self.logger.error(f"Ошибка при предсказании: {str(e)}")
            raise
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Предсказание вероятностей классов.
        
        Args:
            X: Обработанные данные для предсказания
            
        Returns:
            np.ndarray: Массив вероятностей для каждого класса
            
        Raises:
            ValueError: Если модель не загружена
        """
        if not self.is_loaded or self.model is None:
            raise ValueError("Модель не загружена")
        
        try:
            probabilities = self.model.predict_proba(X)
            return probabilities
            
        except Exception as e:
            self.logger.error(f"Ошибка при предсказании вероятностей: {str(e)}")
            raise
    
    def predict_single(self, patient_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Предсказание для одного пациента.
        
        Args:
            patient_data: Словарь с данными пациента
            
        Returns:
            Dict[str, Any]: Результат предсказания с вероятностью
        """
        # Импортируем здесь чтобы избежать циклических импортов
        from .data_processor import DataProcessor
        
        # Валидация входных данных
        self._validate_input_data(patient_data)
        
        # Создание DataFrame из данных пациента
        df = pd.DataFrame([patient_data])
        
        # Предобработка данных (без обучения селектора)
        processor = DataProcessor()
        processed_df = processor.preprocess(df, fit_selector=False)
        
        # Получение предсказания и вероятности
        prediction = self.predict(processed_df)[0]
        probabilities = self.predict_proba(processed_df)[0]
        
        return {
            'prediction': int(prediction),
            'probability': float(probabilities[1])  # Вероятность высокого риска
        }
    
    def predict_batch(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        Пакетное предсказание для множества пациентов.
        
        Args:
            df: DataFrame с данными пациентов (должен содержать колонку 'id')
            
        Returns:
            List[Dict[str, Any]]: Список результатов предсказаний
            
        Raises:
            ValueError: Если отсутствует колонка 'id'
        """
        if 'id' not in df.columns:
            raise ValueError("DataFrame должен содержать колонку 'id'")
        
        # Импортируем здесь чтобы избежать циклических импортов
        from .data_processor import DataProcessor
        
        # Сохраняем ID для результата
        ids = df['id'].copy()
        
        # Предобработка данных (без обучения селектора)
        processor = DataProcessor()
        processed_df = processor.preprocess(df, fit_selector=False)
        
        # Получение предсказаний и вероятностей
        predictions = self.predict(processed_df)
        probabilities = self.predict_proba(processed_df)
        
        # Формирование результата
        results = []
        for i, (patient_id, pred, prob) in enumerate(zip(ids, predictions, probabilities)):
            results.append({
                'id': patient_id,
                'prediction': int(pred),
                'probability': float(prob[1])  # Вероятность высокого риска
            })
        
        return results
    
    def _validate_input_data(self, data: Dict[str, Any]) -> None:
        """
        Валидация входных данных пациента.
        
        Args:
            data: Данные пациента
            
        Raises:
            ValueError: При некорректных данных
        """
        required_fields = [
            'Age', 'BMI', 'Diabetes', 'Systolic blood pressure', 
            'Diastolic blood pressure', 'Triglycerides', 'Sedentary Hours Per Day',
            'Exercise Hours Per Week', 'Cholesterol', 'Income', 'CK-MB', 'Troponin'
        ]
        
        missing_fields = [field for field in required_fields if field not in data]
        if missing_fields:
            raise ValueError(f"Отсутствуют обязательные поля: {missing_fields}")
        
        # Проверка типов данных
        numeric_fields = [
            'Age', 'BMI', 'Diabetes', 'Systolic blood pressure', 
            'Diastolic blood pressure', 'Triglycerides', 'Sedentary Hours Per Day',
            'Exercise Hours Per Week', 'Cholesterol', 'Income', 'CK-MB', 'Troponin'
        ]
        
        for field in numeric_fields:
            if field in data:
                try:
                    float(data[field])
                except (ValueError, TypeError):
                    raise ValueError(f"Неправильный тип данных для поля '{field}': ожидается число")
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Получение информации о модели.
        
        Returns:
            Dict[str, Any]: Информация о модели
        """
        return {
            'model_type': 'CatBoost Classifier',
            'is_loaded': self.is_loaded,
            'model_path': self.model_path,
            'description': 'Модель для предсказания риска сердечного приступа из тетрадки'
        }