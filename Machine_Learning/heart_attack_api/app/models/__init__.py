"""
Модели для предсказания риска сердечного приступа
"""

from .predictor import HeartAttackPredictor
from .data_processor import DataProcessor

__all__ = ['HeartAttackPredictor', 'DataProcessor']