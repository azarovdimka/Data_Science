"""
Класс для предобработки данных пациентов.
"""

import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import LabelEncoder
from typing import List, Optional, Dict, Any
import logging

class DataProcessor:
    """
    Класс для предобработки данных о пациентах перед предсказанием.
    """
    
    def __init__(self):
        """Инициализация процессора данных."""
        self.feature_names = None
        self.label_encoders = {}
        self.is_fitted = False
        
        # Настройка логирования
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Ожидаемые признаки (на основе анализа notebook)
        self.expected_features = [
            'Age', 'Cholesterol', 'Heart rate', 'Diabetes', 'Family History',
            'Smoking', 'Obesity', 'Alcohol Consumption', 'Exercise Hours Per Week',
            'Diet', 'Previous Heart Problems', 'Medication Use', 'Stress Level',
            'Sedentary Hours Per Day', 'Income', 'BMI', 'Triglycerides',
            'Physical Activity Days Per Week', 'Sleep Hours Per Day',
            'Blood sugar', 'CK-MB', 'Troponin', 'Gender',
            'Systolic blood pressure', 'Diastolic blood pressure'
        ]
    
    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Предобработка данных.
        
        Args:
            df: Исходный DataFrame
            
        Returns:
            Обработанный DataFrame
        """
        try:
            self.logger.info("Начало предобработки данных...")
            
            # Создание копии данных
            processed_df = df.copy()
            
            # Удаление ненужных столбцов
            processed_df = self._remove_unnecessary_columns(processed_df)
            
            # Обработка пропущенных значений
            processed_df = self._handle_missing_values(processed_df)
            
            # Обработка категориальных признаков
            processed_df = self._encode_categorical_features(processed_df)
            
            # Проверка и добавление отсутствующих признаков
            processed_df = self._ensure_all_features(processed_df)
            
            # Упорядочивание столбцов
            processed_df = self._reorder_columns(processed_df)
            
            self.logger.info("Предобработка данных завершена")
            return processed_df
            
        except Exception as e:
            self.logger.error(f"Ошибка при предобработке данных: {str(e)}")
            raise
    
    def _remove_unnecessary_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Удаление ненужных столбцов."""
        columns_to_remove = ['Unnamed: 0', 'id', 'Heart Attack Risk (Binary)']
        
        for col in columns_to_remove:
            if col in df.columns:
                df = df.drop(col, axis=1)
                self.logger.info(f"Удален столбец: {col}")
        
        return df
    
    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Обработка пропущенных значений."""
        # Заполнение пропусков медианой для числовых признаков
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_columns:
            if df[col].isnull().sum() > 0:
                median_value = df[col].median()
                df[col] = df[col].fillna(median_value)
                self.logger.info(f"Заполнены пропуски в столбце {col} медианой: {median_value}")
        
        # Заполнение пропусков модой для категориальных признаков
        categorical_columns = df.select_dtypes(include=['object']).columns
        
        for col in categorical_columns:
            if df[col].isnull().sum() > 0:
                mode_value = df[col].mode()[0] if not df[col].mode().empty else 'Unknown'
                df[col] = df[col].fillna(mode_value)
                self.logger.info(f"Заполнены пропуски в столбце {col} модой: {mode_value}")
        
        return df
    
    def _encode_categorical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Кодирование категориальных признаков."""
        # Обработка столбца Gender
        if 'Gender' in df.columns:
            # Обработка случаев, когда Gender уже закодирован как числа
            df['Gender'] = df['Gender'].astype(str)
            
            # Замена числовых значений на текстовые
            gender_mapping = {'1.0': 'Male', '0.0': 'Female', '1': 'Male', '0': 'Female'}
            df['Gender'] = df['Gender'].replace(gender_mapping)
            
            # Кодирование: Male = 1, Female = 0
            if 'Gender' not in self.label_encoders:
                self.label_encoders['Gender'] = LabelEncoder()
                # Фиксированное кодирование для воспроизводимости
                self.label_encoders['Gender'].fit(['Female', 'Male'])
            
            df['Gender'] = self.label_encoders['Gender'].transform(df['Gender'])
            self.logger.info("Закодирован столбец Gender")
        
        return df
    
    def _ensure_all_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Проверка наличия всех необходимых признаков."""
        missing_features = []
        
        for feature in self.expected_features:
            if feature not in df.columns:
                missing_features.append(feature)
        
        if missing_features:
            self.logger.warning(f"Отсутствующие признаки: {missing_features}")
            
            # Добавление отсутствующих признаков с значениями по умолчанию
            for feature in missing_features:
                if feature in ['Gender']:
                    df[feature] = 0  # Female по умолчанию
                elif feature in ['Diet', 'Stress Level', 'Physical Activity Days Per Week']:
                    df[feature] = df[feature].median() if feature in df.columns else 0
                else:
                    df[feature] = 0.0
                
                self.logger.info(f"Добавлен отсутствующий признак {feature} со значением по умолчанию")
        
        return df
    
    def _reorder_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Упорядочивание столбцов согласно ожидаемому порядку."""
        # Фильтрация только существующих столбцов
        available_features = [f for f in self.expected_features if f in df.columns]
        
        # Добавление любых дополнительных столбцов
        extra_columns = [col for col in df.columns if col not in available_features]
        final_columns = available_features + extra_columns
        
        return df[final_columns]
    
    def get_feature_names(self) -> List[str]:
        """Получение списка имен признаков."""
        return self.expected_features.copy()
    
    def validate_input(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Валидация входных данных.
        
        Args:
            df: DataFrame для валидации
            
        Returns:
            Словарь с результатами валидации
        """
        validation_results = {
            'is_valid': True,
            'errors': [],
            'warnings': []
        }
        
        try:
            # Проверка на пустой DataFrame
            if df.empty:
                validation_results['is_valid'] = False
                validation_results['errors'].append("DataFrame пуст")
                return validation_results
            
            # Проверка типов данных
            for col in df.columns:
                if col in ['Gender']:
                    continue  # Категориальный признак
                
                if col in df.select_dtypes(include=[np.number]).columns:
                    # Проверка на отрицательные значения для определенных признаков
                    if col in ['Age', 'BMI', 'Cholesterol'] and (df[col] < 0).any():
                        validation_results['warnings'].append(f"Отрицательные значения в {col}")
                    
                    # Проверка на выбросы (значения больше 1 для нормализованных данных)
                    if col not in ['Diet', 'Stress Level', 'Physical Activity Days Per Week'] and (df[col] > 1).any():
                        validation_results['warnings'].append(f"Значения больше 1 в {col} (возможно, данные не нормализованы)")
            
            # Проверка обязательных столбцов
            required_columns = ['Age', 'Gender', 'Cholesterol', 'Heart rate']
            missing_required = [col for col in required_columns if col not in df.columns]
            
            if missing_required:
                validation_results['is_valid'] = False
                validation_results['errors'].append(f"Отсутствуют обязательные столбцы: {missing_required}")
            
            self.logger.info(f"Валидация завершена. Валидность: {validation_results['is_valid']}")
            return validation_results
            
        except Exception as e:
            validation_results['is_valid'] = False
            validation_results['errors'].append(f"Ошибка валидации: {str(e)}")
            return validation_results
    
    def get_sample_data(self) -> Dict[str, Any]:
        """
        Получение примера данных для тестирования API.
        
        Returns:
            Словарь с примером данных пациента
        """
        return {
            "Age": 0.45,
            "Cholesterol": 0.65,
            "Heart rate": 0.055,
            "Diabetes": 1.0,
            "Family History": 0.0,
            "Smoking": 1.0,
            "Obesity": 0.0,
            "Alcohol Consumption": 1.0,
            "Exercise Hours Per Week": 0.5,
            "Diet": 1,
            "Previous Heart Problems": 0.0,
            "Medication Use": 1.0,
            "Stress Level": 7.0,
            "Sedentary Hours Per Day": 0.4,
            "Income": 0.6,
            "BMI": 0.7,
            "Triglycerides": 0.3,
            "Physical Activity Days Per Week": 3.0,
            "Sleep Hours Per Day": 0.5,
            "Blood sugar": 0.25,
            "CK-MB": 0.048,
            "Troponin": 0.037,
            "Gender": "Male",
            "Systolic blood pressure": 0.45,
            "Diastolic blood pressure": 0.5
        }