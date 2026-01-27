import pandas as pd
import numpy as np
import json
import os
from pathlib import Path
from catboost import CatBoostClassifier
import logging
from .data_processor import DataProcessor

logger = logging.getLogger(__name__)

class HeartAttackPredictor:
    def __init__(self):
        """Инициализация предиктора с оптимизированной моделью"""
        self.model = None
        self.data_processor = DataProcessor()
        self.is_loaded = False
        self.optimal_threshold = 0.5  # Дефолтный порог
        
        # Пути к файлам модели
        self.model_path = Path("models/heart_attack_model.cbm")
        self.threshold_path = Path("models/optimal_threshold.pkl")
        
        try:
            self._load_model()
            logger.info("Predictor инициализирован успешно")
        except Exception as e:
            logger.warning(f"Не удалось загрузить модель: {e}")
            logger.info("Создаю модель-заглушку для демонстрации")
            self._create_dummy_model()

    def _load_model(self):
        """Загрузка обученной модели и оптимального порога"""
        if self.model_path.exists():
            self.model = CatBoostClassifier()
            self.model.load_model(str(self.model_path))
            
            # Загрузка оптимального порога
            if self.threshold_path.exists():
                import joblib
                self.optimal_threshold = joblib.load(self.threshold_path)
                logger.info(f"Оптимальный порог загружен: {self.optimal_threshold}")
            else:
                logger.warning(f"Файл порога не найден: {self.threshold_path}, использую дефолтный 0.5")
            
            self.is_loaded = True
            logger.info(f"Модель загружена из {self.model_path}")
        else:
            raise FileNotFoundError(f"Файл модели не найден: {self.model_path}")

    def _create_dummy_model(self):
        """Создание модели-заглушки для демонстрации"""
        self.model = CatBoostClassifier(
            iterations=10,
            depth=3,
            learning_rate=0.1,
            verbose=False
        )
        
        # Создаем фиктивные данные для обучения заглушки
        np.random.seed(42)
        feature_names = self.data_processor.get_feature_names()
        dummy_data = pd.DataFrame({
            feature: np.random.uniform(0, 1, 100) for feature in feature_names
        })
        dummy_target = np.random.randint(0, 2, 100)
        
        self.model.fit(dummy_data, dummy_target)
        self.is_loaded = True
        logger.info("Создана модель-заглушка для демонстрации")



    def predict_single(self, data):
        """Предсказание для одного образца с оптимальным порогом"""
        try:
            # Преобразуем в DataFrame
            df = pd.DataFrame([data])
            
            # Используем полную предобработку из DataProcessor
            processed_df = self.data_processor.preprocess(df, fit_selector=False)
            
            # Получаем вероятности
            probabilities = self.model.predict_proba(processed_df)[0]
            probability_positive = float(probabilities[1])
            
            # Применяем оптимальный порог для медицинского диагноза
            prediction = 1 if probability_positive >= self.optimal_threshold else 0
            
            return {
                'prediction': int(prediction),
                'probability': probability_positive
            }
            
        except Exception as e:
            logger.error(f"Ошибка при предсказании для одного образца: {e}")
            raise
    
    def predict_batch(self, df):
        """Предсказание для батча данных с оптимальным порогом"""
        try:
            # Сохраняем id если есть
            ids = df['id'].tolist() if 'id' in df.columns else list(range(len(df)))
            
            # Используем полную предобработку из DataProcessor
            processed_df = self.data_processor.preprocess(df, fit_selector=False)
            
            # Получаем вероятности
            probabilities = self.model.predict_proba(processed_df)
            
            # Формируем результат с оптимальным порогом
            results = []
            for i, prob in enumerate(probabilities):
                probability_positive = float(prob[1])
                prediction = 1 if probability_positive >= self.optimal_threshold else 0
                
                results.append({
                    'id': ids[i],
                    'prediction': int(prediction),
                    'probability': probability_positive
                })
            
            return results
            
        except Exception as e:
            logger.error(f"Ошибка при батчевом предсказании: {e}")
            raise
    
    def get_model_info(self):
        """Получение информации о модели"""
        return {
            'model_type': 'CatBoost Classifier',
            'is_loaded': self.is_loaded,
            'model_path': str(self.model_path),
            'optimal_threshold': self.optimal_threshold,
            'threshold_path': str(self.threshold_path),
            'features_count': len(self.data_processor.get_feature_names()) if self.data_processor.selected_features else 'unknown'
        }
    
    def get_feature_importance(self):
        """Получение важности признаков"""
        if self.is_loaded and hasattr(self.model, 'feature_importances_'):
            feature_names = self.data_processor.get_feature_names()
            importance_dict = dict(zip(feature_names, self.model.feature_importances_))
            return importance_dict
        else:
            return {feature: 1.0 for feature in self.data_processor.get_feature_names()}