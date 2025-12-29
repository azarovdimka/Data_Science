"""
Препроцессор данных на основе ml_analysis.ipynb
Включает AudioProcessor и DataPreprocessor для полной обработки данных
"""

import pandas as pd
import numpy as np
import re
import io
import requests
import pymorphy3
from sklearn.base import BaseEstimator, TransformerMixin
from collections import Counter
from mutagen.mp3 import MP3
from typing import Dict, Any

class AudioProcessor(BaseEstimator, TransformerMixin):
    """Класс для обработки аудиофайлов"""
    
    def __init__(self, timeout=30):
        self.timeout = timeout
        self.counter = 0
        self.total_files = 0
        
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X_transformed = X.copy()
        self.counter = 0
        self.total_files = len(X_transformed)
        
        if 'Ссылка на оригинальный файл запис' in X_transformed.columns:
            print("Начинаем скачивать аудиофайлы...")
            X_transformed['Длина_файла'] = X_transformed['Ссылка на оригинальный файл запис'].apply(
                self._get_audio_duration
            )
        else:
            X_transformed['Длина_файла'] = 0
            
        return X_transformed
    
    def _get_audio_duration(self, url):
        """Получение длительности аудиофайла"""
        self.counter += 1
        
        try:
            response = requests.get(url, timeout=self.timeout)
            response.raise_for_status()
            audio_file = MP3(io.BytesIO(response.content))
            duration = audio_file.info.length if audio_file.info.length else 0
            
            print(f"{self.counter}/{self.total_files} Длина файла {duration / 60:.2f} мин")
            return duration
            
        except Exception as e:
            print(f"{self.counter}/{self.total_files} Ошибка загрузки аудио: {type(e).__name__}")
            return 0

class DataPreprocessor(BaseEstimator, TransformerMixin):
    """Основной препроцессор данных"""
    
    def __init__(self):
        self.morph = None
        self.fitted = False
    
    def fit(self, X, y=None):
        if self.morph is None:
            self.morph = pymorphy3.MorphAnalyzer()
        self.fitted = True
        return self
    
    def transform(self, X):
        if not self.fitted:
            raise ValueError("Transformer must be fitted before transform")
        
        X_transformed = X.copy()
        
        # Очистка текста
        X_transformed = self._clean_text_columns(X_transformed)
        
        # НЕ удаляем строки, а заполняем пустые ответы
        empty_mask = ~X_transformed['Транскрибация ответа'].str.contains(r'[а-яёa-z]', case=False, na=False)
        X_transformed.loc[empty_mask, 'Транскрибация ответа'] = 'пустой ответ'
        
        # Анализ речи
        speech_analysis = X_transformed['Транскрибация ответа'].apply(self._analyze_speech_quality)
        
        # Извлечение признаков речи
        for feature in ['Качество_речи', 'Лексическое_разнообразие', 'Словарный_запас', 
                       'noun_ratio', 'verb_ratio', 'adjective_ratio', 'NOUN', 'VERB', 'ADJF']:
            X_transformed[feature] = speech_analysis.apply(lambda x: x.get(feature, 0))
        
        # Создание дополнительных признаков
        X_transformed = self._create_advanced_features(X_transformed)
        X_transformed = self._create_interaction_features(X_transformed)
        
        # Удаление ненужных колонок
        columns_to_drop = ['Id экзамена', 'Id вопроса', 'Ссылка на оригинальный файл запис', 
                          'Картинка из вопроса']
        X_transformed = X_transformed.drop(columns=[col for col in columns_to_drop if col in X_transformed.columns])
        
        return X_transformed
    
    def _clean_text_columns(self, df):
        """Очистка текстовых колонок"""
        for col in ['Текст вопроса', 'Транскрибация ответа']:
            if col in df.columns:
                df[col] = df[col].apply(self._clean_text)
        return df
    
    def _clean_text(self, text):
        """Очистка отдельного текста"""
        if pd.isna(text):
            return text
        
        text = str(text)
        text = re.sub(r'<[^>]+>', '', text)  # HTML теги
        text = text.replace('\n', ' ')        # Переносы строк
        text = re.sub(r'\s+', ' ', text)      # Множественные пробелы
        text = text.strip()                   # Пробелы в начале/конце
        
        return text
    
    def _analyze_speech_quality(self, text):
        """Анализ качества речи"""
        if pd.isna(text) or text == '':
            return self._get_default_analysis()
        
        # Очистка и токенизация
        words = re.findall(r'\b\w+\b', str(text).lower())
        
        if not words:
            return self._get_default_analysis()
        
        # Морфологический анализ
        pos_counts = Counter()
        unique_lemmas = set()
        
        for word in words:
            parsed = self.morph.parse(word)[0]
            
            if parsed.tag.POS:
                pos_counts[parsed.tag.POS] += 1
            
            unique_lemmas.add(parsed.normal_form)
        
        total_words = len(words)
        
        # Расчет метрик
        lexical_diversity = len(unique_lemmas) / total_words if total_words > 0 else 0
        noun_ratio = pos_counts.get('NOUN', 0) / total_words
        verb_ratio = pos_counts.get('VERB', 0) / total_words
        adjective_ratio = pos_counts.get('ADJF', 0) / total_words
        
        complexity_score = self._calculate_complexity(pos_counts, total_words)
        
        quality_score = (
            lexical_diversity + verb_ratio + noun_ratio + 
            adjective_ratio + complexity_score
        )
        
        return {
            'Качество_речи': 190 - (quality_score * 100),
            'Лексическое_разнообразие': lexical_diversity,
            'Словарный_запас': len(unique_lemmas),
            'noun_ratio': noun_ratio,
            'verb_ratio': verb_ratio,
            'adjective_ratio': adjective_ratio,
            'NOUN': pos_counts.get('NOUN', 0),
            'VERB': pos_counts.get('VERB', 0),
            'ADJF': pos_counts.get('ADJF', 0),
        }
    
    def _calculate_complexity(self, pos_counts, total_words):
        """Расчет сложности речи"""
        complexity_weights = {
            'ADJF': 1.6, 'ADVB': 1.8, 'GRND': 1.9, 'PREP': 1.2,
            'CONJ': 1.1, 'VERB': 1.4, 'INFN': 1.3, 'PRTF': 1.5, 'PRTS': 1.5
        }
        
        complexity = 0
        for pos, count in pos_counts.items():
            weight = complexity_weights.get(pos, 1.0)
            complexity += (count / total_words) * weight
        
        return min(complexity, 1)
    
    def _create_advanced_features(self, df):
        """Создание продвинутых признаков"""
        # Пересечение вопроса и ответа
        def text_overlap_ratio(row):
            question = str(row['Текст вопроса']).lower()
            answer = str(row['Транскрибация ответа']).lower()
            
            question_words = set(question.split())
            answer_words = set(answer.split())
            
            if len(question_words) == 0:
                return 0
            
            overlap = len(question_words.intersection(answer_words))
            return overlap / len(question_words)
        
        df['question_overlap_ratio'] = df.apply(text_overlap_ratio, axis=1)
        df['Рассуждение'] = (df['question_overlap_ratio'] >= 0.5).astype(int)
        
        # Длина и скорость
        df['Длина_ответа'] = df['Транскрибация ответа'].str.len()
        df['length_ratio'] = df['Длина_ответа'] / df['Длина_ответа'].mean()
        df['length_factor'] = np.where(df['length_ratio'] > 1, 1, -1)
        
        # Скорость речи (только если есть длина файла)
        if 'Длина_файла' in df.columns:
            df['Скорость_речи'] = df['Длина_ответа'] / (df['Длина_файла'] + 1e-6)
        else:
            df['Скорость_речи'] = 0
        
        # Уникальные слова и свобода речи
        df['Уникальных_слов'] = df['Транскрибация ответа'].apply(
            lambda x: len(set(str(x).split()))
        )
        df['Свобода_речи'] = df['Уникальных_слов'] / (df['Длина_ответа'] + 1e-6)
        
        # Общий балл качества
        df['quality_score'] = (
            -df['Рассуждение'] * 1.5 +
            df['length_factor'] * 1.5 +
            df['Скорость_речи'] * 1.5
        ) + 5
        
        return df
    
    def _create_interaction_features(self, df):
        """Создание признаков взаимодействия"""
        df['quality_vocab_interaction'] = df['quality_score'] * df['Словарный_запас'] * 2
        df['length_speed_interaction'] = df['Длина_ответа'] * df['Скорость_речи'] * 2
        df['noun_adjf_interaction'] = df['NOUN'] * df['ADJF'] * 2
        df['diversity_unique_interaction'] = df['Лексическое_разнообразие'] * df['Уникальных_слов'] * 2
        
        return df
    
    def _get_default_analysis(self):
        """Значения по умолчанию для пустого текста"""
        return {
            'Качество_речи': 0, 'Лексическое_разнообразие': 0, 'Словарный_запас': 0,
            'noun_ratio': 0, 'verb_ratio': 0, 'adjective_ratio': 0,
            'NOUN': 0, 'VERB': 0, 'ADJF': 0
        }