#!/usr/bin/env python3
"""
Скрипт для создания файла с отобранными признаками.
"""

import joblib
import os

# Создаем список отобранных признаков из тетрадки
selected_features = [
    'bmi_diabetes',
    'pressure_product', 
    'lifestyle_score',
    'trig_sedentary',
    'exercise_age1',
    'bmi_exercise1',
    'Systolic blood pressure',
    'cardiac_markers2',
    'chol_income'
]

# Путь к файлу
features_path = "models/selected_features.pkl"

# Создаем директорию если не существует
os.makedirs(os.path.dirname(features_path), exist_ok=True)

# Сохраняем признаки
joblib.dump(selected_features, features_path)

print(f"Файл с признаками создан: {features_path}")
print(f"Количество признаков: {len(selected_features)}")
print("Признаки:")
for i, feature in enumerate(selected_features, 1):
    print(f"{i:2d}. {feature}")