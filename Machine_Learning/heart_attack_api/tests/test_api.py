"""
Тесты для API endpoints.
"""

import pytest
from fastapi.testclient import TestClient
import pandas as pd
import tempfile
import os
from app.main import app


client = TestClient(app)


class TestAPI:
    
    def test_root_endpoint(self):
        """Тест главной страницы."""
        response = client.get("/")
        assert response.status_code == 200
        assert "text/html" in response.headers["content-type"]
    
    def test_health_endpoint(self):
        """Тест health check."""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
    
    def test_predict_single_valid(self):
        """Тест предсказания для одного пациента с валидными данными."""
        patient_data = {
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
        
        response = client.post("/predict", json=patient_data)
        assert response.status_code == 200
        
        data = response.json()
        assert "prediction" in data
        assert "probability" in data
        assert data["prediction"] in [0, 1]
        assert 0 <= data["probability"] <= 1
    
    def test_predict_single_invalid_data(self):
        """Тест предсказания с невалидными данными."""
        invalid_data = {
            "Age": "invalid",  # Строка вместо числа
            "Gender": "Unknown"  # Неизвестный пол
        }
        
        response = client.post("/predict", json=invalid_data)
        assert response.status_code == 422  # Validation error
    
    def test_predict_batch_valid_csv(self):
        """Тест пакетного предсказания с валидным CSV."""
        # Создаем временный CSV файл
        test_data = {
            'id': [1, 2, 3],
            'Age': [0.45, 0.67, 0.23],
            'Cholesterol': [0.65, 0.4, 0.9],
            'Heart rate': [0.055, 0.08, 0.03],
            'Diabetes': [1.0, 0.0, 1.0],
            'Family History': [0.0, 1.0, 0.0],
            'Smoking': [1.0, 0.0, 1.0],
            'Obesity': [0.0, 1.0, 0.0],
            'Alcohol Consumption': [1.0, 0.0, 1.0],
            'Exercise Hours Per Week': [0.5, 0.3, 0.7],
            'Diet': [1, 0, 1],
            'Previous Heart Problems': [0.0, 1.0, 0.0],
            'Medication Use': [1.0, 0.0, 1.0],
            'Stress Level': [7.0, 5.0, 8.0],
            'Sedentary Hours Per Day': [0.4, 0.6, 0.3],
            'Income': [0.6, 0.8, 0.2],
            'BMI': [0.7, 0.5, 0.8],
            'Triglycerides': [0.3, 0.8, 0.2],
            'Physical Activity Days Per Week': [3.0, 2.0, 4.0],
            'Sleep Hours Per Day': [0.5, 0.6, 0.4],
            'Blood sugar': [0.25, 0.15, 0.35],
            'CK-MB': [0.048, 0.02, 0.1],
            'Troponin': [0.037, 0.01, 0.08],
            'Gender': ['Male', 'Female', 'Male'],
            'Systolic blood pressure': [0.45, 0.6, 0.3],
            'Diastolic blood pressure': [0.5, 0.4, 0.7]
        }
        
        df = pd.DataFrame(test_data)
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            df.to_csv(f.name, index=False)
            csv_path = f.name
        
        try:
            with open(csv_path, 'rb') as f:
                response = client.post(
                    "/predict/batch",
                    files={"file": ("test.csv", f, "text/csv")}
                )
            
            assert response.status_code == 200
            
            data = response.json()
            assert "predictions" in data
            assert len(data["predictions"]) == 3
            
            for pred in data["predictions"]:
                assert "id" in pred
                assert "prediction" in pred
                assert "probability" in pred
                
        finally:
            os.unlink(csv_path)
    
    def test_predict_batch_invalid_file(self):
        """Тест пакетного предсказания с невалидным файлом."""
        # Создаем файл с неправильным форматом
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("This is not a CSV file")
            txt_path = f.name
        
        try:
            with open(txt_path, 'rb') as f:
                response = client.post(
                    "/predict/batch",
                    files={"file": ("test.txt", f, "text/plain")}
                )
            
            assert response.status_code == 400
            
        finally:
            os.unlink(txt_path)
    
    def test_predict_batch_missing_columns(self):
        """Тест пакетного предсказания с отсутствующими колонками."""
        # CSV с недостающими колонками
        incomplete_data = pd.DataFrame({
            'id': [1, 2],
            'Age': [0.45, 0.67],
            'Gender': ['Male', 'Female']
            # Много колонок отсутствует
        })
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            incomplete_data.to_csv(f.name, index=False)
            csv_path = f.name
        
        try:
            with open(csv_path, 'rb') as f:
                response = client.post(
                    "/predict/batch",
                    files={"file": ("incomplete.csv", f, "text/csv")}
                )
            
            assert response.status_code == 400
            
        finally:
            os.unlink(csv_path)