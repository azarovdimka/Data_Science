#!/usr/bin/env python3
"""
Скрипт для чтения отобранных признаков модели
"""

import joblib
import json
import os

# Путь к файлу с признаками
features_path = "models/selected_features.pkl"

if os.path.exists(features_path):
    try:
        # Загружаем признаки
        selected_features = joblib.load(features_path)
        
        print("=== ОТОБРАННЫЕ ПРИЗНАКИ ДЛЯ МОДЕЛИ ===")
        print(f"Всего признаков: {len(selected_features)}")
        print("\nСписок признаков:")
        for i, feature in enumerate(selected_features, 1):
            print(f"{i:2d}. {feature}")
        
        # Сохраняем в JSON для удобства
        features_json = {
            "total_features": len(selected_features),
            "selected_features": selected_features,
            "description": "Отобранные признаки после корреляционного анализа и feature engineering"
        }
        
        with open("models/selected_features.json", "w", encoding="utf-8") as f:
            json.dump(features_json, f, ensure_ascii=False, indent=2)
        
        print(f"\n✅ Признаки также сохранены в models/selected_features.json")
        
    except Exception as e:
        print(f"❌ Ошибка при чтении файла: {e}")
else:
    print(f"❌ Файл не найден: {features_path}")