"""
Полный класс для предобработки данных на основе EDA pipeline из notebook.
"""

import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from typing import List, Optional, Dict, Any, Callable, Union
import logging
import joblib
import os

class MistakeCorrector(BaseEstimator, TransformerMixin):
    """Класс для исправления ошибок в данных."""
    
    def __init__(self, columns: List[str], values_dict: Optional[Dict[Any, Any]] = None, 
                 func: Optional[Callable[[Any], Any]] = None, strategy: str = 'dict') -> None:
        """
        Инициализация класса для исправления ошибок в данных.
        
        Args:
            columns: Список столбцов для обработки
            values_dict: Словарь для замены значений
            func: Функция для преобразования
            strategy: Стратегия обработки ('dict' или 'func')
            
        Raises:
            ValueError: При некорректных параметрах
        """
        if not columns:
            raise ValueError("Параметр 'columns' не может быть пустым")
        if strategy not in ['dict', 'func']:
            raise ValueError("strategy должен быть 'dict' или 'func'")
        
        self.values_dict = values_dict or {}
        self.columns = columns
        self.func = func
        self.strategy = strategy
    
    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> 'MistakeCorrector':
        """
        Обучение трансформера (не требуется для данного класса).
        
        Args:
            X: Входные данные
            y: Целевая переменная (не используется)
            
        Returns:
            MistakeCorrector: Ссылка на себя
        """
        return self
    
    def transform(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> pd.DataFrame:
        """
        Применение исправлений к данным.
        
        Args:
            X: Входные данные
            y: Целевая переменная (не используется)
            
        Returns:
            pd.DataFrame: Обработанные данные
            
        Raises:
            RuntimeError: При ошибке применения функции
        """
        df = X.copy()
        
        for col in self.columns:
            if col not in df.columns:
                continue
            
            if self.strategy == 'func' and self.func is not None:
                try:
                    df[col] = df[col].apply(self.func)
                except Exception as e:
                    raise RuntimeError(f"Ошибка при применении функции к столбцу {col}: {e}")
            elif self.strategy == 'dict' and self.values_dict:
                df[col] = df[col].replace(self.values_dict)
        
        return df

class DecimalPointChanger(BaseEstimator, TransformerMixin):
    """Класс для замены разделителя дроби в строковых столбцах."""
    
    def __init__(self, columns: List[str] = None):
        self.columns = columns
    
    def fit(self, X: pd.DataFrame, y=None):
        return self
    
    def transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        df = X.copy()
        cols_to_process = self.columns if self.columns else df.columns
        
        for col in cols_to_process:
            if col in df.columns and df[col].dtype == 'object':
                if df[col].str.contains(',').any():
                    df[col] = df[col].str.replace(',', '.').astype(float)
        
        return df

class OutlierRemover(BaseEstimator, TransformerMixin):
    """Класс для удаления выбросов методом IQR только для train."""
    
    def __init__(self, columns: List[str] = None, factor=1.5):
        self.columns = columns
        self.factor = factor
        self.bounds_dict = {}
    
    def fit(self, X: pd.DataFrame, y=None):
        return self
    
    def transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        # Не удаляем выбросы в медицинских данных
        return X

class DuplicateRemover(BaseEstimator, TransformerMixin):
    """Класс для удаления дубликатов."""
    
    def __init__(self, columns: List[str] = None):
        self.columns = columns
    
    def fit(self, X: pd.DataFrame, y=None):
        return self
    
    def transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        duplicate_count = X.duplicated().sum()
        
        if duplicate_count > 0:
            if self.columns:
                return X.drop_duplicates(subset=self.columns)
            else:
                return X.drop_duplicates()
        
        return X

class MissingValueHandler(BaseEstimator, TransformerMixin):
    """Класс для обработки пропущенных значений."""
    
    def __init__(self, strategy='median', fill_value=None):
        self.strategy = strategy
        self.fill_value = fill_value
        self.fill_values_ = {}
    
    def fit(self, X: pd.DataFrame, y=None):
        if self.strategy == 'median':
            self.fill_values_ = X.select_dtypes(include=[np.number]).median().to_dict()
        elif self.strategy == 'mean':
            self.fill_values_ = X.select_dtypes(include=[np.number]).mean().to_dict()
        elif self.strategy == 'mode':
            self.fill_values_ = {}
            for col in X.columns:
                mode_val = X[col].mode()
                self.fill_values_[col] = mode_val[0] if len(mode_val) > 0 else 0
        
        return self
    
    def transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        df = X.copy()
        
        if self.strategy == 'drop':
            null_count = df.isna().sum().sum()
            if null_count > 0:
                df = df.dropna()
            return df
        
        # Заполнение пропусков
        for col, fill_value in self.fill_values_.items():
            if col in df.columns and df[col].isnull().sum() > 0:
                df[col] = df[col].fillna(fill_value)
        
        return df

class ColumnRemover(BaseEstimator, TransformerMixin):
    """Удаляет лишние колонки."""
    
    def __init__(self, columns: List[str]):
        self.columns = columns
    
    def fit(self, X: pd.DataFrame, y=None):
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        df = X.copy()
        for col in self.columns:
            if col in df.columns:
                df = df.drop(col, axis=1)
        return df

class FloatToIntChanger(BaseEstimator, TransformerMixin):
    """Преобразует дробные значения в целочисленные."""
    
    def __init__(self, columns: List[str], strategy='simple'):
        self.columns = columns
        self.strategy = strategy
    
    def fit(self, X: pd.DataFrame, y=None):
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        df = X.copy()
        
        for col in self.columns:
            if col in df.columns:
                if self.strategy == 'simple':
                    df[col] = df[col].astype('int')
                elif self.strategy == 'multiply':
                    df[col] = (df[col] * 100).astype('int')
        
        return df

class EDAPreprocessor:
    """Основной класс пайплайна для предобработки данных."""
    
    def __init__(self):
        self.steps = []
        self.fitted_transformers = {}
    
    def add_mistake_corrector(self, columns: List[str] = None, values_dict: dict = None, 
                            func: Callable = None, strategy: str = None, 
                            step_name: str = 'Преобразование некорректных данных'):
        mistake_corrector = MistakeCorrector(
            columns=columns, values_dict=values_dict, func=func, strategy=strategy
        )
        self.steps.append((step_name, mistake_corrector))
        return self
    
    def add_column_remover(self, columns: List[str], step_name: str = 'Удаление столбцов'):
        column_remover = ColumnRemover(columns)
        self.steps.append((step_name, column_remover))
        return self
    
    def add_decimal_point_changer(self, columns: List[str] = None, 
                                step_name='Замена запятой на точку'):
        decimal_point_changer = DecimalPointChanger(columns)
        self.steps.append((step_name, decimal_point_changer))
        return self
    
    def add_missing_value_handler(self, strategy='median', fill_value=None, 
                                step_name='Обработка пропущенных значений'):
        missing_handler = MissingValueHandler(strategy=strategy, fill_value=fill_value)
        self.steps.append((step_name, missing_handler))
        return self
    
    def add_drop_duplicates(self, step_name='Удаление дубликатов'):
        duplicate_remover = DuplicateRemover()
        self.steps.append((step_name, duplicate_remover))
        return self
    
    def fit_transform(self, X: pd.DataFrame, name='data', y=None):
        df = X.copy()
        
        for i, (step_name, transformer) in enumerate(self.steps):
            df = transformer.fit_transform(df)
            self.fitted_transformers[step_name] = transformer
        
        return df

class DataProcessor:
    """
    Полный класс для предобработки данных с feature engineering и отбором признаков.
    """
    
    def __init__(self) -> None:
        """
        Инициализация процессора данных.
        """
        self.preprocessor: Optional[EDAPreprocessor] = None
        self.selected_features: Optional[List[str]] = None
        self.label_encoders: Dict[str, Any] = {}
        self.scaler: Optional[StandardScaler] = StandardScaler()  # Инициализируем скейлер
        self.is_fitted: bool = False
        
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Попытка загрузить сохраненный процессор
        self._try_load_saved_processor()
        
        # Создание пайплайна предобработки если не загружен
        if not self.is_fitted:
            self._create_preprocessing_pipeline()
    
    def _try_load_saved_processor(self) -> None:
        """
        Загрузка списка отобранных признаков из тетрадки.
        """
        features_path = "models/selected_features.pkl"
        
        if os.path.exists(features_path):
            try:
                self.selected_features = joblib.load(features_path)
                self.is_fitted = True
                self.logger.info(f"Загружены отобранные признаки: {self.selected_features}")
            except Exception as e:
                self.logger.warning(f"Ошибка загрузки признаков: {e}")
        else:
            self.logger.info("Файл с признаками не найден, использую дефолтные")
    
    def _create_preprocessing_pipeline(self) -> None:
        """Создание пайплайна предобработки точно как в notebook."""
        self.preprocessor = (
            EDAPreprocessor()
            .add_column_remover(columns=['Unnamed: 0', 'id'])  # Удаляем лишние столбцы
            .add_decimal_point_changer()  # Заменяем запятую на точку в дробных числах
            .add_missing_value_handler(strategy='median')  # Обрабатываем пропуски медианой
            .add_drop_duplicates()  # Удаляем дубликаты
        )
        self.logger.info("Пайплайн предобработки создан")
    
    def preprocess(self, df: pd.DataFrame, fit_selector: bool = False) -> pd.DataFrame:
        """
        Полная предобработка данных точно как в notebook.
        
        Args:
            df: Исходный DataFrame
            fit_selector: Обучать ли селектор признаков
            
        Returns:
            Обработанный DataFrame
        """
        try:
            self.logger.info("Начало полной предобработки данных...")
            
            # Сохранение целевой переменной если есть
            target = None
            if 'Heart Attack Risk (Binary)' in df.columns:
                target = df['Heart Attack Risk (Binary)'].copy()
            
            # Проверяем, что preprocessor инициализирован
            if self.preprocessor is None:
                self.logger.warning("Preprocessor не инициализирован, создаем новый")
                self._create_preprocessing_pipeline()
            
            # Базовая предобработка через пайплайн (удаление столбцов, обработка пропусков)
            processed_df = self.preprocessor.fit_transform(df.copy())
            
            # Кодирование категориальных признаков
            processed_df = self._encode_categorical_features(processed_df)
            
            # Feature Engineering - создание новых признаков как в тетрадке
            processed_df = self._create_new_features(processed_df)
            
            # Отбор признаков - только для обучающей выборки
            if fit_selector and target is not None:
                processed_df = self._fit_feature_selection(processed_df, target)
            elif self.selected_features is not None:
                # Применяем обученный селектор к тестовым данным
                processed_df = self._apply_feature_selection(processed_df)
            
            # Масштабирование
            if fit_selector:
                # Обучаем скейлер на обучающих данных
                if self.scaler is None:
                    self.scaler = StandardScaler()
                processed_df = pd.DataFrame(
                    self.scaler.fit_transform(processed_df),
                    columns=processed_df.columns,
                    index=processed_df.index
                )
            elif self.scaler is not None:
                # Применяем обученный скейлер к тестовым данным
                try:
                    processed_df = pd.DataFrame(
                        self.scaler.transform(processed_df),
                        columns=processed_df.columns,
                        index=processed_df.index
                    )
                except Exception as e:
                    self.logger.warning(f"Ошибка при масштабировании: {e}. Пропускаем масштабирование.")
            
            self.is_fitted = True
            self.logger.info(f"Предобработка завершена. Итоговых признаков: {processed_df.shape[1]}")
            return processed_df
            
        except Exception as e:
            self.logger.error(f"Ошибка при предобработке: {str(e)}")
            raise
    
    def _encode_categorical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Кодирование категориальных признаков."""
        if 'Gender' in df.columns:
            df['Gender'] = df['Gender'].astype(str)
            
            # Замена числовых значений на текстовые
            gender_mapping = {'1.0': 'Male', '0.0': 'Female', '1': 'Male', '0': 'Female'}
            df['Gender'] = df['Gender'].replace(gender_mapping)
            
            # Кодирование
            if 'Gender' not in self.label_encoders:
                self.label_encoders['Gender'] = LabelEncoder()
                self.label_encoders['Gender'].fit(['Female', 'Male'])
            
            df['Gender'] = self.label_encoders['Gender'].transform(df['Gender'])
        
        return df
    
    def _create_new_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Создание производных признаков точно как в тетрадке."""
        self.logger.info("Создание производных признаков...")
        
        # Проверка наличия необходимых столбцов
        required_cols = ['BMI', 'Diabetes', 'Systolic blood pressure', 'Diastolic blood pressure',
                        'Triglycerides', 'Sedentary Hours Per Day', 'Exercise Hours Per Week', 
                        'Age', 'Cholesterol', 'Income', 'CK-MB', 'Troponin']
        
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            self.logger.warning(f"Отсутствующие столбцы: {missing_cols}")
            return df
        
        # Создание только тех признаков, которые отобраны в тетрадке
        
        # 1. bmi_diabetes - самый важный признак (corr: 0.419)
        df['bmi_diabetes'] = df['BMI'] * df['Diabetes']
        
        # 2. pressure_product - произведение давлений (corr: 0.065)
        df['pressure_product'] = df['Systolic blood pressure'] * df['Diastolic blood pressure']
        
        # 3. lifestyle_score - оценка образа жизни (corr: 0.045)
        df['lifestyle_score'] = df['Exercise Hours Per Week'] / (df['Sedentary Hours Per Day'] + 0.001)
        
        # 4. trig_sedentary - взаимодействие триглицеридов и малоподвижности (corr: 0.041)
        df['trig_sedentary'] = df['Triglycerides'] * df['Sedentary Hours Per Day']
        
        # 5. exercise_age1 - взаимодействие упражнений и возраста (corr: 0.037)
        df['exercise_age1'] = df['Exercise Hours Per Week'] * df['Age']
        
        # 6. bmi_exercise1 - взаимодействие BMI и упражнений (corr: 0.035)
        df['bmi_exercise1'] = df['BMI'] * df['Exercise Hours Per Week']
        
        # 7. Systolic blood pressure - оставляем как есть (corr: 0.032)
        
        # 8. cardiac_markers2 - комбинация сердечных маркеров (corr: 0.031)
        df['cardiac_markers2'] = df['CK-MB'] + df['Troponin']
        
        # 9. chol_income - взаимодействие холестерина и дохода (corr: 0.030)
        df['chol_income'] = df['Cholesterol'] * df['Income']
        
        self.logger.info("Создано 8 производных признаков как в тетрадке")
        return df
    
    def _fit_feature_selection(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        """Отбор признаков по корреляции с целевой переменной как в тетрадке."""
        self.logger.info("Отбор признаков по корреляции с целевой переменной...")
        
        # 1. Удаление сильно коррелированных признаков между собой
        X_corr_filtered = self._remove_highly_correlated_features(X)
        
        # 2. Отбор по корреляции с целевой переменной (0.03 <= corr <= 0.9)
        target_corr = X_corr_filtered.corrwith(y).abs()
        selected_features = target_corr[(target_corr >= 0.03) & (target_corr <= 0.9)].index.tolist()
        
        if not selected_features:
            self.logger.warning("Не найдено признаков с корреляцией в диапазоне [0.03, 0.9]. Использую все признаки.")
            selected_features = X_corr_filtered.columns.tolist()
        
        self.selected_features = selected_features
        X_selected = X_corr_filtered[selected_features]
        
        self.logger.info(f"Отобрано {len(selected_features)} признаков по корреляции с целевой переменной")
        
        # Логирование топ-15 признаков по корреляции
        feature_corr = [(f, target_corr[f]) for f in selected_features]
        feature_corr.sort(key=lambda x: x[1], reverse=True)
        
        self.logger.info("Топ-15 признаков по корреляции:")
        for i, (feature, corr) in enumerate(feature_corr[:15]):
            self.logger.info(f"{i+1:2d}. {feature:<35} {corr:.4f}")
        
        return X_selected
    
    def _remove_highly_correlated_features(self, X: pd.DataFrame, threshold: float = 0.9) -> pd.DataFrame:
        """Удаление сильно коррелированных признаков как в тетрадке."""
        self.logger.info(f"Удаление признаков с корреляцией > {threshold}...")
        
        # Вычисляем корреляционную матрицу
        corr_matrix = X.corr().abs()
        
        # Находим пары признаков с высокой корреляцией
        upper_tri = corr_matrix.where(
            np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
        )
        
        # Находим признаки для удаления
        to_drop = [column for column in upper_tri.columns if any(upper_tri[column] > threshold)]
        
        if to_drop:
            self.logger.info(f"Удаляю {len(to_drop)} сильно коррелированных признаков: {to_drop[:5]}...")
            X_filtered = X.drop(columns=to_drop)
        else:
            self.logger.info("Сильно коррелированных признаков не найдено")
            X_filtered = X
        
        return X_filtered
    
    def _apply_feature_selection(self, X: pd.DataFrame) -> pd.DataFrame:
        """Применение обученного селектора признаков."""
        if self.selected_features is None:
            self.logger.warning("Селектор признаков не обучен")
            return X
        
        # Для тестовых данных НЕ анализируем корреляцию, просто отбираем нужные признаки
        missing_features = [f for f in self.selected_features if f not in X.columns]
        if missing_features:
            self.logger.warning(f"Отсутствующие признаки: {missing_features}")
            available_features = [f for f in self.selected_features if f in X.columns]
            return X[available_features]
        
        return X[self.selected_features]
    
    def get_feature_names(self) -> List[str]:
        """Получение списка выбранных признаков."""
        if self.selected_features is not None:
            return self.selected_features.copy()
        else:
            # Возвращаем топ-9 признаков из тетрадки
            return [
                'bmi_diabetes', 'pressure_product', 'lifestyle_score', 'trig_sedentary',
                'exercise_age1', 'bmi_exercise1', 'Systolic blood pressure', 
                'cardiac_markers2', 'chol_income'
            ]
    
    def save_processor(self, path: str):
        """Сохранение обученного процессора."""
        processor_data = {
            'preprocessor': self.preprocessor,
            'selected_features': self.selected_features,
            'label_encoders': self.label_encoders,
            'scaler': self.scaler,
            'is_fitted': self.is_fitted
        }
        
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(processor_data, path)
        self.logger.info(f"Процессор сохранен: {path}")
    
    def load_processor(self, path: str):
        """Загрузка обученного процессора."""
        if not os.path.exists(path):
            raise FileNotFoundError(f"Файл процессора не найден: {path}")
        
        processor_data = joblib.load(path)
        
        self.preprocessor = processor_data.get('preprocessor')
        self.selected_features = processor_data.get('selected_features')
        self.label_encoders = processor_data.get('label_encoders', {})
        self.scaler = processor_data.get('scaler')
        self.is_fitted = processor_data.get('is_fitted', False)
        
        self.logger.info(f"Процессор загружен: {path}")
    
    def get_sample_data(self) -> Dict[str, Any]:
        """Получение примера данных для тестирования API."""
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