#!/usr/bin/env python3
"""
Создание файла с оптимальным порогом для медицинской модели.
Порог 0.480 был определен в тетрадке для достижения recall 92%+
"""

import joblib
import os

# Оптимальный порог из тетрадки
OPTIMAL_THRESHOLD = 0.480

# Создание директории если не существует
os.makedirs("models", exist_ok=True)

# Сохранение порога
joblib.dump(OPTIMAL_THRESHOLD, "models/optimal_threshold.pkl")

print(f"Оптимальный порог {OPTIMAL_THRESHOLD} сохранен в models/optimal_threshold.pkl")
print("Этот порог обеспечивает recall 92%+ для медицинской диагностики")