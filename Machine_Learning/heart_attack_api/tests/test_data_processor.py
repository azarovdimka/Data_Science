"""
Тесты для DataProcessor.
"""

import pytest
import pandas as pd
import numpy as np
from app.models.data_processor import DataProcessor


class TestDataProcessor:
    
    def setup_method(self):
        """Настройка для каждого теста."""
        self.processor = DataProcessor()
        
        # Создание тестовых данных
        self.sample_data = {
            'Age': [0.45, 0.67, 0.23],
            'BMI': [0.7, 0.5, 0.8],
            'Diabetes': [1.0, 0.0, 1.0],
            'Systolic blood pressure': [0.45, 0.6, 0.3],
            'Diastolic blood pressure': [0.5, 0.4, 0.7],
            'Triglycerides': [0.3, 0.8, 0.2],
            'Sedentary Hours Per Day': [0.4, 0.6, 0.3],
            'Exercise Hours Per Week': [0.5, 0.3, 0.7],
            'Cholesterol': [0.65, 0.4, 0.9],
            'Income': [0.6, 0.8, 0.2],
            'CK-MB': [0.048, 0.02, 0.1],
            'Troponin': [0.037, 0.01, 0.08],
            'Gender': ['Male', 'Female', 'Male'],
            'Heart Attack Risk (Binary)': [1, 0, 1]
        }
        self.df = pd.DataFrame(self.sample_data)
    
    def test_create_new_features(self):
        """Тест создания производных признаков."""
        processed_df = self.processor._create_new_features(self.df.copy())
        
        # Проверяем, что созданы все 8 производных признаков
        expected_features = [
            'bmi_diabetes', 'pressure_product', 'lifestyle_score', 
            'trig_sedentary', 'exercise_age1', 'bmi_exercise1', 
            'cardiac_markers2', 'chol_income'
        ]
        
        for feature in expected_features:
            assert feature in processed_df.columns, f"Признак {feature} не создан"
        
        # Проверяем корректность вычислений
        assert processed_df['bmi_diabetes'].iloc[0] == self.df['BMI'].iloc[0] * self.df['Diabetes'].iloc[0]
        assert processed_df['pressure_product'].iloc[0] == (
            self.df['Systolic blood pressure'].iloc[0] * self.df['Diastolic blood pressure'].iloc[0]
        )
    
    def test_encode_categorical_features(self):
        """Тест кодирования категориальных признаков."""
        processed_df = self.processor._encode_categorical_features(self.df.copy())
        
        # Проверяем, что Gender закодирован в числа
        assert processed_df['Gender'].dtype in [np.int64, np.int32]
        assert set(processed_df['Gender'].unique()).issubset({0, 1})
    
    def test_preprocess_with_target(self):
        """Тест полной предобработки с целевой переменной."""
        processed_df = self.processor.preprocess(self.df.copy(), fit_selector=True)
        
        # Проверяем, что данные обработаны
        assert processed_df is not None
        assert len(processed_df.columns) > 0
        
        # Проверяем, что селектор обучен
        assert self.processor.selected_features is not None
        assert len(self.processor.selected_features) > 0
    
    def test_preprocess_without_target(self):
        """Тест предобработки без целевой переменной (тестовые данные)."""
        # Удаляем целевую переменную
        test_df = self.df.drop('Heart Attack Risk (Binary)', axis=1)
        
        # Сначала обучаем на данных с целевой переменной
        self.processor.preprocess(self.df.copy(), fit_selector=True)
        
        # Затем применяем к тестовым данным
        processed_df = self.processor.preprocess(test_df, fit_selector=False)
        
        assert processed_df is not None
        assert len(processed_df.columns) > 0
    
    def test_get_feature_names(self):
        """Тест получения имен признаков."""
        # До обучения должны возвращаться дефолтные признаки
        default_features = self.processor.get_feature_names()
        expected_default = [
            'bmi_diabetes', 'pressure_product', 'lifestyle_score', 'trig_sedentary',
            'exercise_age1', 'bmi_exercise1', 'Systolic blood pressure', 
            'cardiac_markers2', 'chol_income'
        ]
        assert default_features == expected_default
        
        # После обучения должны возвращаться выбранные признаки
        self.processor.preprocess(self.df.copy(), fit_selector=True)
        selected_features = self.processor.get_feature_names()
        assert selected_features == self.processor.selected_features
    
    def test_save_load_processor(self, tmp_path):
        """Тест сохранения и загрузки процессора."""
        # Обучаем процессор
        self.processor.preprocess(self.df.copy(), fit_selector=True)
        
        # Сохраняем
        save_path = tmp_path / "test_processor.pkl"
        self.processor.save_processor(str(save_path))
        
        # Создаем новый процессор и загружаем
        new_processor = DataProcessor()
        new_processor.load_processor(str(save_path))
        
        # Проверяем, что состояние восстановлено
        assert new_processor.selected_features == self.processor.selected_features
        assert new_processor.is_fitted == self.processor.is_fitted
    
    def test_get_sample_data(self):
        """Тест получения примера данных."""
        sample = self.processor.get_sample_data()
        
        assert isinstance(sample, dict)
        assert 'Age' in sample
        assert 'Gender' in sample
        assert len(sample) > 20  # Должно быть много признаков