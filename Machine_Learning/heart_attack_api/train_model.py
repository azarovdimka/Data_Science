"""
Скрипт для обучения модели предсказания сердечных приступов.
"""

import pandas as pd
import numpy as np
import sys
import os
from pathlib import Path

# Добавление пути к модулям приложения
sys.path.append(str(Path(__file__).parent))

from app.models.predictor import HeartAttackPredictor
from app.models.data_processor import DataProcessor

def load_data():
    """Загрузка данных для обучения."""
    try:
        # Попытка загрузить данные из разных возможных путей
        possible_paths = [
            'C:/DS/datasets/heart_train.csv',
            './Heart_attacks/datasets/heart_train.csv',
            '../Heart_attacks/datasets/heart_train.csv',
            'data/heart_train.csv'
        ]
        
        train_df = None
        for path in possible_paths:
            if os.path.exists(path):
                train_df = pd.read_csv(path)
                print(f"Данные загружены из: {path}")
                break
        
        if train_df is None:
            raise FileNotFoundError("Файл с обучающими данными не найден")
        
        return train_df
        
    except Exception as e:
        print(f"Ошибка при загрузке данных: {e}")
        raise

def prepare_data(df):
    """Подготовка данных для обучения."""
    # Инициализация процессора данных
    processor = DataProcessor()
    
    # Сохранение целевой переменной
    if 'Heart Attack Risk (Binary)' in df.columns:
        y = df['Heart Attack Risk (Binary)']
    else:
        raise ValueError("Целевая переменная 'Heart Attack Risk (Binary)' не найдена")
    
    # Предобработка признаков
    X = processor.preprocess(df)
    
    print(f"Форма данных после предобработки: {X.shape}")
    print(f"Признаки: {list(X.columns)}")
    print(f"Распределение целевой переменной: {y.value_counts().to_dict()}")
    
    return X, y, processor

def train_model():
    """Основная функция обучения модели."""
    try:
        print("=== Обучение модели предсказания сердечных приступов ===")
        
        # Загрузка данных
        print("\n1. Загрузка данных...")
        train_df = load_data()
        print(f"Загружено {len(train_df)} записей")
        
        # Подготовка данных
        print("\n2. Подготовка данных...")
        X, y, processor = prepare_data(train_df)
        
        # Инициализация и обучение модели
        print("\n3. Обучение модели...")
        predictor = HeartAttackPredictor()
        
        # Обучение
        training_results = predictor.train(X, y, validation_split=0.2)
        
        # Вывод результатов
        print("\n4. Результаты обучения:")
        print(f"Точность на валидации: {training_results['accuracy']:.4f}")
        
        print("\nТоп-10 важных признаков:")
        feature_importance = training_results['feature_importance']
        sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
        
        for i, (feature, importance) in enumerate(sorted_features[:10]):
            print(f"{i+1:2d}. {feature:<30} {importance:.4f}")
        
        # Сохранение модели
        print("\n5. Сохранение модели...")
        model_dir = "models"
        os.makedirs(model_dir, exist_ok=True)
        
        model_path = os.path.join(model_dir, "heart_attack_model.cbm")
        predictor.save_model(model_path)
        
        print(f"Модель сохранена: {model_path}")
        
        # Создание тестового предсказания
        print("\n6. Тестовое предсказание...")
        sample_data = processor.get_sample_data()
        sample_df = pd.DataFrame([sample_data])
        processed_sample = processor.preprocess(sample_df)
        
        prediction = predictor.predict(processed_sample)[0]
        probabilities = predictor.predict_proba(processed_sample)[0]
        
        print(f"Тестовое предсказание: {prediction}")
        print(f"Вероятности: Низкий риск={probabilities[0]:.3f}, Высокий риск={probabilities[1]:.3f}")
        
        print("\n=== Обучение завершено успешно ===")
        
        return predictor, processor
        
    except Exception as e:
        print(f"Ошибка при обучении модели: {e}")
        raise

if __name__ == "__main__":
    train_model()