"""
Класс для предсказания риска сердечного приступа.
"""

import pandas as pd
import numpy as np
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os
from typing import List, Optional, Union
import logging

class HeartAttackPredictor:
    """
    Класс для предсказания риска сердечного приступа с использованием CatBoost.
    """
    
    def __init__(self, model_path: Optional[str] = None):
        """
        Инициализация предиктора.
        
        Args:
            model_path: Путь к сохраненной модели
        """
        self.model = None
        self.feature_names = None
        self.is_trained = False
        self.model_path = model_path or "models/heart_attack_model.cbm"
        
        # Настройка логирования
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Попытка загрузить существующую модель
        if os.path.exists(self.model_path):
            self.load_model()
        else:
            self.model = CatBoostClassifier(
                iterations=1000,
                learning_rate=0.1,
                depth=6,
                loss_function='Logloss',
                eval_metric='Accuracy',
                random_seed=42,
                verbose=False
            )
    
    def train(self, X: pd.DataFrame, y: pd.Series, validation_split: float = 0.2) -> dict:
        """
        Обучение модели.
        
        Args:
            X: Признаки для обучения
            y: Целевая переменная
            validation_split: Доля данных для валидации
            
        Returns:
            Словарь с метриками обучения
        """
        try:
            self.logger.info("Начало обучения модели...")
            
            # Сохранение имен признаков
            self.feature_names = list(X.columns)
            
            # Разделение на обучающую и валидационную выборки
            X_train, X_val, y_train, y_val = train_test_split(
                X, y, test_size=validation_split, random_state=42, stratify=y
            )
            
            # Обучение модели
            self.model.fit(
                X_train, y_train,
                eval_set=(X_val, y_val),
                early_stopping_rounds=100,
                verbose=False
            )
            
            # Предсказания на валидационной выборке
            y_pred = self.model.predict(X_val)
            accuracy = accuracy_score(y_val, y_pred)
            
            self.is_trained = True
            self.logger.info(f"Модель обучена. Точность на валидации: {accuracy:.4f}")
            
            # Сохранение модели
            self.save_model()
            
            return {
                "accuracy": accuracy,
                "feature_importance": dict(zip(self.feature_names, self.model.feature_importances_)),
                "classification_report": classification_report(y_val, y_pred, output_dict=True)
            }
            
        except Exception as e:
            self.logger.error(f"Ошибка при обучении модели: {str(e)}")
            raise
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Предсказание риска сердечного приступа.
        
        Args:
            X: Данные для предсказания
            
        Returns:
            Массив предсказаний (0 - низкий риск, 1 - высокий риск)
        """
        if not self.is_trained and self.model is None:
            raise ValueError("Модель не обучена. Сначала обучите модель или загрузите существующую.")
        
        try:
            # Проверка соответствия признаков
            if self.feature_names and set(X.columns) != set(self.feature_names):
                missing_features = set(self.feature_names) - set(X.columns)
                extra_features = set(X.columns) - set(self.feature_names)
                
                error_msg = ""
                if missing_features:
                    error_msg += f"Отсутствующие признаки: {missing_features}. "
                if extra_features:
                    error_msg += f"Лишние признаки: {extra_features}. "
                
                raise ValueError(error_msg)
            
            # Упорядочивание столбцов согласно обученной модели
            if self.feature_names:
                X = X[self.feature_names]
            
            predictions = self.model.predict(X)
            return predictions.astype(int)
            
        except Exception as e:
            self.logger.error(f"Ошибка при предсказании: {str(e)}")
            raise
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Предсказание вероятностей классов.
        
        Args:
            X: Данные для предсказания
            
        Returns:
            Массив вероятностей для каждого класса
        """
        if not self.is_trained and self.model is None:
            raise ValueError("Модель не обучена.")
        
        try:
            if self.feature_names:
                X = X[self.feature_names]
            
            probabilities = self.model.predict_proba(X)
            return probabilities
            
        except Exception as e:
            self.logger.error(f"Ошибка при предсказании вероятностей: {str(e)}")
            raise
    
    def save_model(self, path: Optional[str] = None) -> None:
        """
        Сохранение модели.
        
        Args:
            path: Путь для сохранения модели
        """
        if not self.is_trained:
            raise ValueError("Модель не обучена.")
        
        save_path = path or self.model_path
        
        try:
            # Создание директории если не существует
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            
            # Сохранение модели CatBoost
            self.model.save_model(save_path)
            
            # Сохранение дополнительной информации
            model_info = {
                'feature_names': self.feature_names,
                'is_trained': self.is_trained
            }
            
            info_path = save_path.replace('.cbm', '_info.pkl')
            joblib.dump(model_info, info_path)
            
            self.logger.info(f"Модель сохранена: {save_path}")
            
        except Exception as e:
            self.logger.error(f"Ошибка при сохранении модели: {str(e)}")
            raise
    
    def load_model(self, path: Optional[str] = None) -> None:
        """
        Загрузка модели.
        
        Args:
            path: Путь к модели
        """
        load_path = path or self.model_path
        
        try:
            if not os.path.exists(load_path):
                raise FileNotFoundError(f"Файл модели не найден: {load_path}")
            
            # Загрузка модели CatBoost
            self.model = CatBoostClassifier()
            self.model.load_model(load_path)
            
            # Загрузка дополнительной информации
            info_path = load_path.replace('.cbm', '_info.pkl')
            if os.path.exists(info_path):
                model_info = joblib.load(info_path)
                self.feature_names = model_info.get('feature_names')
                self.is_trained = model_info.get('is_trained', True)
            else:
                self.is_trained = True
            
            self.logger.info(f"Модель загружена: {load_path}")
            
        except Exception as e:
            self.logger.error(f"Ошибка при загрузке модели: {str(e)}")
            raise
    
    def get_feature_importance(self) -> dict:
        """
        Получение важности признаков.
        
        Returns:
            Словарь с важностью признаков
        """
        if not self.is_trained:
            raise ValueError("Модель не обучена.")
        
        if self.feature_names:
            return dict(zip(self.feature_names, self.model.feature_importances_))
        else:
            return dict(enumerate(self.model.feature_importances_))
    
    def create_predictions_csv(self, X: pd.DataFrame, output_path: str, id_column: str = 'id') -> str:
        """
        Создание CSV файла с предсказаниями в требуемом формате.
        
        Args:
            X: Данные для предсказания
            output_path: Путь для сохранения файла
            id_column: Название колонки с ID
            
        Returns:
            Путь к созданному файлу
        """
        try:
            predictions = self.predict(X)
            
            # Создание DataFrame с результатами
            if id_column in X.columns:
                results_df = pd.DataFrame({
                    'id': X[id_column],
                    'prediction': predictions
                })
            else:
                results_df = pd.DataFrame({
                    'id': range(len(predictions)),
                    'prediction': predictions
                })
            
            # Сохранение в CSV
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            results_df.to_csv(output_path, index=False)
            
            self.logger.info(f"Файл с предсказаниями сохранен: {output_path}")
            return output_path
            
        except Exception as e:
            self.logger.error(f"Ошибка при создании CSV файла: {str(e)}")
            raise