"""
Сервис машинного обучения для оценки ответов

Загрузка готовой модели
Предобработка текста (очистка HTML, нормализация)
Извлечение признаков (длина текста, количество слов и т.д.)
Предсказание оценок с учетом максимальных баллов по вопросам
"""

import pandas as pd
import numpy as np
import pickle
import re
import io
import requests
from typing import List
from collections import Counter
import os
import warnings
from mutagen.mp3 import MP3
import pymorphy3


warnings.filterwarnings('ignore', message='.*version.*')
warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')

class AudioProcessor:
    """Отдельный класс для обработки аудиофайлов"""
    
    def __init__(self, timeout=300):
        self.timeout = timeout
        self.counter = 0
        self.total_files = 0
        
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

class MLService:
    """Сервис для загрузки готовой модели и предсказаний"""
    
    def __init__(self):
        self.model = None
        self.morph = None
        self.semantic_analyzer = None
        self.linguistic_analyzer = None
        self.question_max_scores = {1: 1, 2: 2, 3: 1, 4: 2}
        self._initialize_morph()
        self._initialize_analyzers()
        self.load_model()
    
    def _initialize_morph(self):
        """Инициализация морфологического анализатора"""
        if pymorphy3:
            try:
                self.morph = pymorphy3.MorphAnalyzer()
                print("Морфологический анализатор pymorphy3 загружен")
            except Exception as e:
                print(f"Ошибка загрузки pymorphy3: {e}")
                self.morph = None
        else:
            print("pymorphy3 не установлен, используется упрощенный анализ")
    
    
    def load_model(self):
        """Загрузка обученной модели"""
        # Пробуем разные пути к модели
        possible_paths = [
            '/var/www/rus_exam_site/ml_models/models/evaluate_exam_model.pkl',
            'ml_models/models/evaluate_exam_model.pkl',
            os.path.join(os.getcwd(), 'ml_models', 'models', 'evaluate_exam_model.pkl'),
            os.path.join(os.path.dirname(__file__), '..', '..', 'ml_models', 'models', 'evaluate_exam_model.pkl')
        ]
        
        model_path = None
        for path in possible_paths:
            if os.path.exists(path):
                model_path = path
                break
        
        if model_path:
            try:
                with open(model_path, 'rb') as f:
                    self.model = pickle.load(f)
                print(f"Модель загружена из {model_path}")
            except Exception as e:
                print(f"Ошибка загрузки модели: {e}")
                self.model = None
        else:
            print(f"Модель не найдена по путям: {possible_paths}")
            print(f"Текущая директория: {os.getcwd()}")
            self.model = None
    
    def _clean_text(self, text):
        """Очистка отдельного текста"""

        print("Очищаю текст ответа от лишних")
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
        print("Анализирую качеств речи")
        if pd.isna(text) or text == '':
            return self._get_default_analysis()
        
        # Очистка и токенизация
        words = re.findall(r'\b\w+\b', str(text).lower())
        
        if not words:
            return self._get_default_analysis()
        
        # Морфологический анализ
        pos_counts = Counter()
        unique_lemmas = set()
        
        if self.morph:
            for word in words:
                parsed = self.morph.parse(word)[0]
                
                if parsed.tag.POS:
                    pos_counts[parsed.tag.POS] += 1
                
                unique_lemmas.add(parsed.normal_form)
        else:
            # Упрощенный анализ без pymorphy3
            for word in words:
                unique_lemmas.add(word)
                # Простая эвристика для определения частей речи
                if word.endswith(('ый', 'ая', 'ое', 'ые')):
                    pos_counts['ADJF'] += 1
                elif word.endswith(('ть', 'ет', 'ит', 'ут', 'ют')):
                    pos_counts['VERB'] += 1
                else:
                    pos_counts['NOUN'] += 1
        
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
        print("Рассчитываю сложность речи")
        complexity_weights = {
            'ADJF': 1.6, 'ADVB': 1.8, 'GRND': 1.9, 'PREP': 1.2,
            'CONJ': 1.1, 'VERB': 1.4, 'INFN': 1.3, 'PRTF': 1.5, 'PRTS': 1.5
        }
        
        complexity = 0
        for pos, count in pos_counts.items():
            weight = complexity_weights.get(pos, 1.0)
            complexity += (count / total_words) * weight
        
        return min(complexity, 1)
    
    def _get_default_analysis(self):
        """Значения по умолчанию для пустого текста"""
        return {
            'Качество_речи': 0, 'Лексическое_разнообразие': 0, 'Словарный_запас': 0,
            'noun_ratio': 0, 'verb_ratio': 0, 'adjective_ratio': 0,
            'NOUN': 0, 'VERB': 0, 'ADJF': 0
        }
    
    def _create_advanced_features(self, df):
        """Создание продвинутых признаков"""
        print("Создаю и рассчитываю дополнительные признаки для оценки качества речи")
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
        
        # Длина ответа в символах
        df['Длина_ответа'] = df['Транскрибация ответа'].str.len()
        df['length_ratio'] = df['Длина_ответа'] / (df['Длина_ответа'].mean() + 1e-6)
        df['length_factor'] = np.where(df['length_ratio'] > 1, 1, -1)
        
        # Скорость речи = символы в секунду
        df['Скорость_речи'] = df['Длина_ответа'] / (df['Длина_файла'] + 1e-6)
        
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
    
    def _add_semantic_features(self, df):
        """Добавление семантических признаков"""
        print("Добавляю семантические признаки")
        
        if self.semantic_analyzer is None:
            # Заглушки для семантических признаков
            df['semantic_similarity'] = 0.5
            df['lexical_overlap'] = 0.5
            df['relevance_score'] = 0.5
            return df
        
        semantic_scores = []
        for idx, row in df.iterrows():
            question = str(row.get('Текст вопроса', ''))
            answer = str(row.get('Транскрибация ответа', ''))
            
            try:
                relevance = self.semantic_analyzer.analyze_answer_relevance(question, answer)
                semantic_scores.append(relevance)
            except Exception as e:
                print(f"Ошибка семантического анализа: {e}")
                semantic_scores.append({
                    'semantic_similarity': 0.5,
                    'lexical_overlap': 0.5,
                    'relevance_score': 0.5
                })
        
        # Добавляем признаки
        for i, scores in enumerate(semantic_scores):
            df.loc[df.index[i], 'semantic_similarity'] = scores.get('semantic_similarity', 0.5)
            df.loc[df.index[i], 'lexical_overlap'] = scores.get('lexical_overlap', 0.5)
            df.loc[df.index[i], 'relevance_score'] = scores.get('relevance_score', 0.5)
        
        return df
    
    def _add_linguistic_features(self, df):
        """Добавление лингвистических признаков"""
        print("Добавляю лингвистические признаки")
        
        if self.linguistic_analyzer is None:
            # Заглушки для лингвистических признаков
            df['grammar_score'] = 0.7
            df['coherence_score'] = 0.7
            df['terminology_score'] = 0.6
            df['complexity_score'] = 0.6
            return df
        
        linguistic_scores = []
        for idx, row in df.iterrows():
            answer = str(row.get('Транскрибация ответа', ''))
            
            try:
                analysis = self.linguistic_analyzer.analyze_text(answer)
                linguistic_scores.append(analysis)
            except Exception as e:
                print(f"Ошибка лингвистического анализа: {e}")
                linguistic_scores.append({
                    'grammar_score': 0.7,
                    'coherence_score': 0.7,
                    'terminology_score': 0.6,
                    'complexity_score': 0.6
                })
        
        # Добавляем признаки
        for i, scores in enumerate(linguistic_scores):
            df.loc[df.index[i], 'grammar_score'] = scores.get('grammar_score', 0.7)
            df.loc[df.index[i], 'coherence_score'] = scores.get('coherence_score', 0.7)
            df.loc[df.index[i], 'terminology_score'] = scores.get('terminology_score', 0.6)
            df.loc[df.index[i], 'complexity_score'] = scores.get('complexity_score', 0.6)
        
        return df
    
    async def _preprocess_data(self, df, progress_callback=None):
        """Полная предобработка данных"""
        X_transformed = df.copy()
        
        # Обработка аудиофайлов (если есть ссылки) - создает колонку Длина_файла
        if 'Ссылка на оригинальный файл запис' in X_transformed.columns:
            if progress_callback: await progress_callback(10, "Обработка аудиофайлов")
            audio_processor = AudioProcessor()
            X_transformed = audio_processor.transform(X_transformed)
        
        # Очистка текста
        if progress_callback: await progress_callback(20, 'Очистка текста')
        for col in ['Текст вопроса', 'Транскрибация ответа']:
            if col in X_transformed.columns:
                X_transformed[col] = X_transformed[col].apply(self._clean_text)
        
        # Фильтрация пустых ответов
        if progress_callback: await progress_callback(30, 'Фильтрация пустых ответов')
        empty_mask = ~X_transformed['Транскрибация ответа'].str.contains(r'[а-яёa-z]', case=False, na=False)
        X_transformed.loc[empty_mask, 'Транскрибация ответа'] = 'пустой ответ'
        
        # Анализ речи - создание морфологических признаков
        if progress_callback: await progress_callback(50, 'Анализ речи')
        speech_analysis = X_transformed['Транскрибация ответа'].apply(self._analyze_speech_quality)
        
        # Извлечение признаков речи
        if progress_callback: await progress_callback(70, "Создание признаков речи")
        for feature in ['Качество_речи', 'Лексическое_разнообразие', 'Словарный_запас', 
                       'noun_ratio', 'verb_ratio', 'adjective_ratio', 'NOUN', 'VERB', 'ADJF']:
            X_transformed[feature] = speech_analysis.apply(lambda x: x.get(feature, 0))
        
        # Создание дополнительных признаков
        if progress_callback: await progress_callback(75, "Создание дополнительных признаков")
        X_transformed = self._create_advanced_features(X_transformed)
        X_transformed = self._create_interaction_features(X_transformed)
        
        # Добавление семантических и лингвистических признаков
        if progress_callback: await progress_callback(85, "Добавление семантических признаков")
        X_transformed = self._add_semantic_features(X_transformed)
        
        if progress_callback: await progress_callback(90, "Добавление лингвистических признаков")
        X_transformed = self._add_linguistic_features(X_transformed)
        
        
        if progress_callback: await progress_callback(95, f"Предобработка завершена. Колонок: {len(X_transformed.columns)}")
        return X_transformed
    
    async def predict_scores(self, df: pd.DataFrame, progress_callback=None) -> List[float]:
        """Предсказание оценок для DataFrame"""
        try:
            if self.model is None:
                raise ValueError("Модель не загружена. Проверьте наличие файла модели.")
            
            print("Модель загружена - выполняем предсказание")
            
            # Полная предобработка данных
            df_processed = await self._preprocess_data(df, progress_callback or self._send_progress)
            
            # Предсказание
            predictions = self.model.predict(df_processed)
            
            # Обрезка предсказаний
            clipped_predictions = []
            for i, pred in enumerate(predictions):
                question_num = df.iloc[i]['№ вопроса']
                max_score = self.question_max_scores.get(question_num, 2)
                clipped_pred = max(0, min(pred, max_score))
                clipped_predictions.append(round(clipped_pred))
            
            return clipped_predictions
            
        except Exception as e:
            import traceback
            error_details = traceback.format_exc()
            raise ValueError(f"Ошибка предсказания: {str(e)}\n\nПодробности:\n{error_details}")
    
    async def predict_single(self, question_number: int, question_text: str, transcription: str) -> int:
        """Предсказание оценки для одного ответа"""
        temp_df = pd.DataFrame({
            '№ вопроса': [question_number],
            'Текст вопроса': [question_text],
            'Транскрибация ответа': [transcription]
        })
        
        predictions = await self.predict_scores(temp_df)
        return predictions[0]
    
    def get_max_score(self, question_number: int) -> int:
        """Получение максимальной оценки для вопроса"""
        return self.question_max_scores.get(question_number, 2)
    
    def is_model_loaded(self) -> bool:
        """Проверка загружена ли модель"""
        return self.model is not None
    
    async def _send_progress(self, percent: int, message: str):
        """Отправка прогресса через WebSocket"""
        try:
            from .websocket import progress_manager
            await progress_manager.send_progress(percent, message)
        except:
            # Если WebSocket недоступен, просто выводим в консоль
            print(f"{percent}%: {message}")