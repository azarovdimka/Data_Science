"""
Тесты для ML сервиса
"""

import pytest
import pandas as pd
import numpy as np
import sys
import os

# Добавляем путь к проекту
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from backend.services.ml_service import MLService

class TestMLService:
    """Тесты для MLService"""
    
    @pytest.fixture
    def ml_service(self):
        """Создание экземпляра MLService для тестов"""
        return MLService()
    
    @pytest.fixture
    def sample_data(self):
        """Создание тестовых данных"""
        return pd.DataFrame({
            'Id экзамена': ['1', '2', '3', '4'],
            'Id вопроса': ['q1', 'q2', 'q3', 'q4'],
            '№ вопроса': [1, 2, 3, 4],
            'Текст вопроса': [
                'Расскажите о себе',
                'Опишите картинку',
                'Начните диалог',
                'Ответьте на вопросы'
            ],
            'Транскрибация ответа': [
                'Меня зовут Иван, я студент',
                'На картинке изображена семья',
                'Здравствуйте, как дела',
                'Я живу в Москве уже пять лет'
            ],
            'Оценка экзаменатора': [1, 2, 1, 2]
        })
    
    def test_preprocess_text(self, ml_service):
        """Тест предобработки текста"""
        # Тест с HTML тегами
        html_text = "<p>Привет, <strong>мир</strong>!</p>"
        result = ml_service.preprocess_text(html_text)
        assert result == "привет мир"
        
        # Тест с пустым текстом
        empty_result = ml_service.preprocess_text("")
        assert empty_result == ""
        
        # Тест с None
        none_result = ml_service.preprocess_text(None)
        assert none_result == ""
        
        # Тест с обычным текстом
        normal_text = "Привет, как дела?"
        normal_result = ml_service.preprocess_text(normal_text)
        assert normal_result == "привет как дела"
    
    def test_extract_features(self, ml_service, sample_data):
        """Тест извлечения признаков"""
        features = ml_service.extract_features(sample_data)
        
        # Проверяем, что все признаки присутствуют
        expected_features = [
            'transcription_length', 'word_count', 'question_number',
            'avg_word_length', 'question_words_in_answer', 'sentence_count'
        ]
        
        for feature in expected_features:
            assert feature in features.columns
        
        # Проверяем размер
        assert len(features) == len(sample_data)
        
        # Проверяем типы данных
        assert features['word_count'].dtype in [np.int64, np.float64]
        assert features['question_number'].dtype in [np.int64, np.float64]
    
    def test_train_model(self, ml_service, sample_data):
        """Тест обучения модели"""
        # Обучаем модель
        ml_service.train_model(sample_data)
        
        # Проверяем, что модель и векторизатор созданы
        assert ml_service.model is not None
        assert ml_service.vectorizer is not None
        
        # Проверяем, что модель может делать предсказания
        features = ml_service.extract_features(sample_data)
        text_features = ml_service.vectorizer.transform(sample_data['processed_transcription'])
        
        assert text_features.shape[0] == len(sample_data)
    
    @pytest.mark.asyncio
    async def test_predict_scores(self, ml_service, sample_data):
        """Тест предсказания оценок"""
        # Сначала обучаем модель
        ml_service.train_model(sample_data)
        
        # Делаем предсказания
        predictions = await ml_service.predict_scores(sample_data)
        
        # Проверяем результат
        assert len(predictions) == len(sample_data)
        assert all(isinstance(pred, (int, float)) for pred in predictions)
        
        # Проверяем, что оценки в допустимых пределах
        for i, pred in enumerate(predictions):
            question_num = sample_data.iloc[i]['№ вопроса']
            max_score = ml_service.question_max_scores.get(question_num, 2)
            assert 0 <= pred <= max_score
    
    @pytest.mark.asyncio
    async def test_predict_single(self, ml_service, sample_data):
        """Тест предсказания одного ответа"""
        # Обучаем модель
        ml_service.train_model(sample_data)
        
        # Предсказываем одну оценку
        prediction = await ml_service.predict_single(
            question_number=1,
            question_text="Расскажите о себе",
            transcription="Меня зовут Анна, я работаю учителем"
        )
        
        # Проверяем результат
        assert isinstance(prediction, (int, float))
        assert 0 <= prediction <= 1  # Для вопроса 1 максимум 1 балл
    
    def test_get_max_score(self, ml_service):
        """Тест получения максимальной оценки"""
        assert ml_service.get_max_score(1) == 1
        assert ml_service.get_max_score(2) == 2
        assert ml_service.get_max_score(3) == 1
        assert ml_service.get_max_score(4) == 2
        assert ml_service.get_max_score(5) == 2  # По умолчанию
    
    def test_is_model_loaded(self, ml_service, sample_data):
        """Тест проверки загрузки модели"""
        # Изначально модель не загружена
        assert not ml_service.is_model_loaded()
        
        # После обучения модель загружена
        ml_service.train_model(sample_data)
        assert ml_service.is_model_loaded()
    
    def test_edge_cases(self, ml_service):
        """Тест граничных случаев"""
        # Пустой DataFrame
        empty_df = pd.DataFrame(columns=[
            'Id экзамена', 'Id вопроса', '№ вопроса', 
            'Текст вопроса', 'Транскрибация ответа', 'Оценка экзаменатора'
        ])
        
        features = ml_service.extract_features(empty_df)
        assert len(features) == 0
        
        # DataFrame с пустыми транскрипциями
        empty_transcription_df = pd.DataFrame({
            'Id экзамена': ['1'],
            'Id вопроса': ['q1'],
            '№ вопроса': [1],
            'Текст вопроса': ['Вопрос'],
            'Транскрибация ответа': [''],
            'Оценка экзаменатора': [0]
        })
        
        features = ml_service.extract_features(empty_transcription_df)
        assert len(features) == 1
        assert features.iloc[0]['word_count'] == 0
    
    def test_feature_consistency(self, ml_service, sample_data):
        """Тест консистентности признаков"""
        # Извлекаем признаки дважды
        features1 = ml_service.extract_features(sample_data)
        features2 = ml_service.extract_features(sample_data)
        
        # Проверяем, что результаты одинаковые
        pd.testing.assert_frame_equal(features1, features2)
    
    @pytest.mark.parametrize("question_num,expected_max", [
        (1, 1), (2, 2), (3, 1), (4, 2), (999, 2)
    ])
    def test_max_scores_parametrized(self, ml_service, question_num, expected_max):
        """Параметризованный тест максимальных оценок"""
        assert ml_service.get_max_score(question_num) == expected_max

if __name__ == "__main__":
    pytest.main([__file__])