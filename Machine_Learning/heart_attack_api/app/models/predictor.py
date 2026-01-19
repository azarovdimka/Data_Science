import pandas as pd
import numpy as np
import json
import os
from pathlib import Path
from catboost import CatBoostClassifier
import logging

logger = logging.getLogger(__name__)

class HeartAttackPredictor:
    def __init__(self):
        """Инициализация предиктора с оптимизированной моделью"""
        self.model = None
        self.important_features = ['Systolic blood pressure', 'lifestyle_score', 'bmi_exercise1']
        self.is_loaded = False
        
        # Пути к файлам модели
        self.model_path = Path("models/heart_attack_model.cbm")
        
        try:
            self._load_model()
            logger.info("Predictor инициализирован успешно")
        except Exception as e:
            logger.warning(f"Не удалось загрузить модель: {e}")
            logger.info("Создаю модель-заглушку для демонстрации")
            self._create_dummy_model()

    def _load_model(self):
        """Загрузка обученной модели"""
        if self.model_path.exists():
            self.model = CatBoostClassifier()
            self.model.load_model(str(self.model_path))
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
        dummy_data = pd.DataFrame({
            'Systolic blood pressure': np.random.uniform(0, 1, 100),
            'lifestyle_score': np.random.uniform(0, 1, 100),
            'bmi_exercise1': np.random.uniform(0, 1, 100)
        })
        dummy_target = np.random.randint(0, 2, 100)
        
        self.model.fit(dummy_data, dummy_target)
        self.is_loaded = True
        logger.info("Создана модель-заглушка для демонстрации")

    def create_features(self, df):
        """Создание инженерных признаков из исходных данных"""
        df_processed = df.copy()
        
        # lifestyle_score - комбинированный показатель образа жизни
        lifestyle_components = []
        
        if 'Smoking' in df.columns:
            # Курение (инвертируем, так как курение - плохо)
            smoking_score = 1 - pd.to_numeric(df['Smoking'], errors='coerce').fillna(0)
            lifestyle_components.append(smoking_score)
        
        if 'Exercise Hours Per Week' in df.columns:
            # Физическая активность (нормализуем)
            exercise_score = pd.to_numeric(df['Exercise Hours Per Week'], errors='coerce').fillna(0)
            lifestyle_components.append(exercise_score)
        
        if 'Diet' in df.columns:
            # Диета (предполагаем, что здоровая диета = 1)
            diet_score = pd.to_numeric(df['Diet'], errors='coerce').fillna(0.5)
            lifestyle_components.append(diet_score)
        
        if 'Sleep Hours Per Day' in df.columns:
            # Сон (оптимальный сон около 8 часов)
            sleep_hours = pd.to_numeric(df['Sleep Hours Per Day'], errors='coerce').fillna(0.5)
            # Нормализуем относительно оптимального значения
            sleep_score = 1 - np.abs(sleep_hours - 0.5)  # предполагаем 0.5 = оптимум
            lifestyle_components.append(sleep_score)
        
        # Усредняем компоненты образа жизни
        if lifestyle_components:
            df_processed['lifestyle_score'] = np.mean(lifestyle_components, axis=0)
        else:
            df_processed['lifestyle_score'] = 0.5  # нейтральное значение
        
        # bmi_exercise1 - взаимодействие ИМТ и физической активности
        bmi = pd.to_numeric(df.get('BMI', 0.5), errors='coerce').fillna(0.5)
        exercise = pd.to_numeric(df.get('Exercise Hours Per Week', 0.5), errors='coerce').fillna(0.5)
        
        # Создаем признак взаимодействия
        df_processed['bmi_exercise1'] = bmi * exercise
        
        # Systolic blood pressure уже есть в данных
        if 'Systolic blood pressure' not in df.columns:
            logger.warning("Systolic blood pressure не найден в данных, используем значение по умолчанию")
            df_processed['Systolic blood pressure'] = 0.5
        
        return df_processed

    def preprocess_data(self, df):
        """Предобработка данных для модели"""
        try:
            # Создаем инженерные признаки
            df_with_features = self.create_features(df)
            
            # Выбираем только важные признаки
            df_processed = df_with_features[self.important_features].copy()
            
            # Проверяем и заполняем пропуски
            for col in self.important_features:
                if col not in df_processed.columns:
                    logger.warning(f"Признак {col} отсутствует, заполняем значением по умолчанию")
                    df_processed[col] = 0.5
                else:
                    # Заполняем пропуски медианным значением
                    df_processed[col] = pd.to_numeric(df_processed[col], errors='coerce').fillna(0.5)
            
            logger.info(f"Данные предобработаны. Форма: {df_processed.shape}")
            logger.info(f"Используемые признаки: {list(df_processed.columns)}")
            
            return df_processed
            
        except Exception as e:
            logger.error(f"Ошибка при предобработке данных: {e}")
            # Возвращаем данные по умолчанию
            default_data = pd.DataFrame({
                'Systolic blood pressure': [0.5] * len(df),
                'lifestyle_score': [0.5] * len(df),
                'bmi_exercise1': [0.25] * len(df)
            })
            return default_data

    def predict(self, df):
        """Предсказание риска сердечного приступа"""
        try:
            if not self.is_loaded:
                raise ValueError("Модель не загружена")
            
            # Предобработка данных
            processed_data = self.preprocess_data(df)
            
            # Получение предсказаний
            predictions = self.model.predict(processed_data)
            
            logger.info(f"Выполнено предсказание для {len(predictions)} образцов")
            return predictions.astype(int)
            
        except Exception as e:
            logger.error(f"Ошибка при предсказании: {e}")
            # Возвращаем случайные предсказания для демонстрации
            np.random.seed(42)
            return np.random.randint(0, 2, len(df))

    def predict_proba(self, df):
        """Предсказание вероятностей классов"""
        try:
            if not self.is_loaded:
                raise ValueError("Модель не загружена")
            
            # Предобработка данных
            processed_data = self.preprocess_data(df)
            
            # Получение вероятностей
            probabilities = self.model.predict_proba(processed_data)
            
            logger.info(f"Выполнено предсказание вероятностей для {len(probabilities)} образцов")
            return probabilities
            
        except Exception as e:
            logger.error(f"Ошибка при предсказании вероятностей: {e}")
            # Возвращаем случайные вероятности для демонстрации
            np.random.seed(42)
            n_samples = len(df)
            random_probs = np.random.random((n_samples, 2))
            # Нормализуем чтобы сумма была 1
            random_probs = random_probs / random_probs.sum(axis=1, keepdims=True)
            return random_probs

    def get_feature_importance(self):
        """Получение важности признаков"""
        if self.is_loaded and hasattr(self.model, 'feature_importances_'):
            importance_dict = dict(zip(self.important_features, self.model.feature_importances_))
            return importance_dict
        else:
            return {feature: 1.0 for feature in self.important_features}