"""
Тесты для Predictor класса.
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch
from app.models.predictor import Predictor


class TestPredictor:
    
    def setup_method(self):
        """Настройка для каждого теста."""
        self.predictor = Predictor()
        
        # Мокаем модель
        self.mock_model = Mock()
        self.mock_model.predict.return_value = np.array([1, 0, 1])
        self.mock_model.predict_proba.return_value = np.array([[0.2, 0.8], [0.7, 0.3], [0.1, 0.9]])
        
        # Мокаем процессор
        self.mock_processor = Mock()
        self.mock_processor.preprocess.return_value = pd.DataFrame({
            'feature1': [0.5, 0.3, 0.8],
            'feature2': [0.2, 0.9, 0.1]
        })
        
        self.predictor.model = self.mock_model
        self.predictor.processor = self.mock_processor
        self.predictor.is_loaded = True
    
    def test_predict_single_patient(self):
        """Тест предсказания для одного пациента."""
        patient_data = {
            'Age': 0.45,
            'Gender': 'Male',
            'BMI': 0.7
        }
        
        result = self.predictor.predict_single(patient_data)
        
        assert 'prediction' in result
        assert 'probability' in result
        assert result['prediction'] in [0, 1]
        assert 0 <= result['probability'] <= 1
        
        # Проверяем, что методы были вызваны
        self.mock_processor.preprocess.assert_called_once()
        self.mock_model.predict.assert_called_once()
        self.mock_model.predict_proba.assert_called_once()
    
    def test_predict_batch(self):
        """Тест пакетного предсказания."""
        df = pd.DataFrame({
            'id': [1, 2, 3],
            'Age': [0.45, 0.67, 0.23],
            'Gender': ['Male', 'Female', 'Male'],
            'BMI': [0.7, 0.5, 0.8]
        })
        
        results = self.predictor.predict_batch(df)
        
        assert len(results) == 3
        for result in results:
            assert 'id' in result
            assert 'prediction' in result
            assert 'probability' in result
            assert result['prediction'] in [0, 1]
            assert 0 <= result['probability'] <= 1
    
    def test_predict_without_loaded_model(self):
        """Тест предсказания без загруженной модели."""
        predictor = Predictor()  # Новый экземпляр без загруженной модели
        
        with pytest.raises(ValueError, match="Модель не загружена"):
            predictor.predict_single({'Age': 0.5})
    
    def test_predict_batch_missing_id_column(self):
        """Тест пакетного предсказания без колонки id."""
        df = pd.DataFrame({
            'Age': [0.45, 0.67],
            'Gender': ['Male', 'Female']
        })
        
        with pytest.raises(ValueError, match="отсутствует колонка 'id'"):
            self.predictor.predict_batch(df)
    
    @patch('app.models.predictor.joblib.load')
    @patch('os.path.exists')
    def test_load_model_success(self, mock_exists, mock_joblib_load):
        """Тест успешной загрузки модели."""
        mock_exists.return_value = True
        mock_joblib_load.return_value = self.mock_model
        
        predictor = Predictor()
        predictor.load_model()
        
        assert predictor.is_loaded
        assert predictor.model is not None
    
    @patch('os.path.exists')
    def test_load_model_file_not_found(self, mock_exists):
        """Тест загрузки несуществующей модели."""
        mock_exists.return_value = False
        
        predictor = Predictor()
        
        with pytest.raises(FileNotFoundError):
            predictor.load_model()
    
    def test_validate_input_data_valid(self):
        """Тест валидации корректных данных."""
        valid_data = {
            'Age': 0.45,
            'Gender': 'Male',
            'BMI': 0.7,
            'Diabetes': 1.0
        }
        
        # Не должно вызывать исключение
        self.predictor._validate_input_data(valid_data)
    
    def test_validate_input_data_missing_required(self):
        """Тест валидации с отсутствующими обязательными полями."""
        invalid_data = {
            'Age': 0.45
            # Отсутствуют другие обязательные поля
        }
        
        with pytest.raises(ValueError, match="Отсутствуют обязательные поля"):
            self.predictor._validate_input_data(invalid_data)
    
    def test_validate_input_data_invalid_types(self):
        """Тест валидации с неправильными типами данных."""
        invalid_data = {
            'Age': 'invalid_age',  # Строка вместо числа
            'Gender': 'Male',
            'BMI': 0.7
        }
        
        with pytest.raises(ValueError, match="Неправильный тип данных"):
            self.predictor._validate_input_data(invalid_data)