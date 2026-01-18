"""
Скрипт для тестирования API предсказания сердечных приступов.
"""

import requests
import json
import pandas as pd
import os
from typing import Dict, Any

class HeartAttackAPITester:
    """Класс для тестирования API предсказания сердечных приступов."""
    
    def __init__(self, base_url: str = "http://localhost:8001"):
        """
        Инициализация тестера.
        
        Args:
            base_url: Базовый URL API
        """
        self.base_url = base_url
        self.session = requests.Session()
    
    def test_health_check(self) -> bool:
        """Тест проверки состояния API."""
        try:
            response = self.session.get(f"{self.base_url}/health")
            
            if response.status_code == 200:
                data = response.json()
                print("✅ Health check: OK")
                print(f"   Status: {data.get('status')}")
                print(f"   Message: {data.get('message')}")
                return True
            else:
                print(f"❌ Health check failed: {response.status_code}")
                return False
                
        except Exception as e:
            print(f"❌ Health check error: {e}")
            return False
    
    def test_json_prediction(self) -> bool:
        """Тест предсказания через JSON."""
        try:
            # Тестовые данные пациента
            test_data = {
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
            
            response = self.session.post(
                f"{self.base_url}/predict/json",
                json=test_data,
                headers={"Content-Type": "application/json"}
            )
            
            if response.status_code == 200:
                data = response.json()
                print("✅ JSON prediction: OK")
                print(f"   Prediction: {data.get('prediction')}")
                print(f"   Risk level: {data.get('risk_level')}")
                print(f"   Message: {data.get('message')}")
                return True
            else:
                print(f"❌ JSON prediction failed: {response.status_code}")
                print(f"   Response: {response.text}")
                return False
                
        except Exception as e:
            print(f"❌ JSON prediction error: {e}")
            return False
    
    def test_csv_prediction(self, csv_file_path: str = None) -> bool:
        """
        Тест предсказания через CSV файл.
        
        Args:
            csv_file_path: Путь к CSV файлу для тестирования
        """
        try:
            # Создание тестового CSV файла если не предоставлен
            if csv_file_path is None or not os.path.exists(csv_file_path):
                csv_file_path = self._create_test_csv()
            
            with open(csv_file_path, 'rb') as f:
                files = {'file': ('test_data.csv', f, 'text/csv')}
                response = self.session.post(f"{self.base_url}/predict/csv", files=files)
            
            if response.status_code == 200:
                data = response.json()
                print("✅ CSV prediction: OK")
                print(f"   Total predictions: {data.get('total_predictions')}")
                print(f"   Message: {data.get('message')}")
                
                # Показать несколько примеров предсказаний
                predictions = data.get('predictions', {})
                print("   Sample predictions:")
                for i, (id_val, pred) in enumerate(list(predictions.items())[:3]):
                    risk = "High risk" if pred == 1 else "Low risk"
                    print(f"     ID {id_val}: {pred} ({risk})")
                
                return True
            else:
                print(f"❌ CSV prediction failed: {response.status_code}")
                print(f"   Response: {response.text}")
                return False
                
        except Exception as e:
            print(f"❌ CSV prediction error: {e}")
            return False
    
    def test_model_info(self) -> bool:
        """Тест получения информации о модели."""
        try:
            response = self.session.get(f"{self.base_url}/model/info")
            
            if response.status_code == 200:
                data = response.json()
                print("✅ Model info: OK")
                print(f"   Model type: {data.get('model_type')}")
                print(f"   Version: {data.get('version')}")
                print(f"   Features count: {len(data.get('features', []))}")
                return True
            else:
                print(f"❌ Model info failed: {response.status_code}")
                return False
                
        except Exception as e:
            print(f"❌ Model info error: {e}")
            return False
    
    def _create_test_csv(self) -> str:
        """Создание тестового CSV файла."""
        test_data = [
            {
                "id": 1,
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
            },
            {
                "id": 2,
                "Age": 0.25,
                "Cholesterol": 0.35,
                "Heart rate": 0.045,
                "Diabetes": 0.0,
                "Family History": 1.0,
                "Smoking": 0.0,
                "Obesity": 1.0,
                "Alcohol Consumption": 0.0,
                "Exercise Hours Per Week": 0.8,
                "Diet": 2,
                "Previous Heart Problems": 1.0,
                "Medication Use": 0.0,
                "Stress Level": 3.0,
                "Sedentary Hours Per Day": 0.2,
                "Income": 0.8,
                "BMI": 0.4,
                "Triglycerides": 0.6,
                "Physical Activity Days Per Week": 5.0,
                "Sleep Hours Per Day": 0.7,
                "Blood sugar": 0.15,
                "CK-MB": 0.025,
                "Troponin": 0.020,
                "Gender": "Female",
                "Systolic blood pressure": 0.35,
                "Diastolic blood pressure": 0.4
            }
        ]
        
        df = pd.DataFrame(test_data)
        csv_path = "test_data.csv"
        df.to_csv(csv_path, index=False)
        
        print(f"📁 Created test CSV file: {csv_path}")
        return csv_path
    
    def run_all_tests(self, csv_file_path: str = None) -> Dict[str, bool]:
        """
        Запуск всех тестов.
        
        Args:
            csv_file_path: Путь к CSV файлу для тестирования
            
        Returns:
            Словарь с результатами тестов
        """
        print("🧪 Starting API tests...")
        print("=" * 50)
        
        results = {}
        
        # Тест проверки состояния
        print("\n1. Testing health check...")
        results['health_check'] = self.test_health_check()
        
        # Тест информации о модели
        print("\n2. Testing model info...")
        results['model_info'] = self.test_model_info()
        
        # Тест JSON предсказания
        print("\n3. Testing JSON prediction...")
        results['json_prediction'] = self.test_json_prediction()
        
        # Тест CSV предсказания
        print("\n4. Testing CSV prediction...")
        results['csv_prediction'] = self.test_csv_prediction(csv_file_path)
        
        # Сводка результатов
        print("\n" + "=" * 50)
        print("📊 Test Results Summary:")
        
        passed = sum(results.values())
        total = len(results)
        
        for test_name, result in results.items():
            status = "✅ PASSED" if result else "❌ FAILED"
            print(f"   {test_name}: {status}")
        
        print(f"\nOverall: {passed}/{total} tests passed")
        
        if passed == total:
            print("🎉 All tests passed!")
        else:
            print("⚠️  Some tests failed. Check the API server.")
        
        return results

def main():
    """Основная функция для запуска тестов."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Test Heart Attack Prediction API")
    parser.add_argument("--url", default="http://localhost:8000", help="API base URL")
    parser.add_argument("--csv", help="Path to CSV file for testing")
    
    args = parser.parse_args()
    
    tester = HeartAttackAPITester(args.url)
    results = tester.run_all_tests(args.csv)
    
    # Очистка тестовых файлов
    if os.path.exists("test_data.csv"):
        os.remove("test_data.csv")
        print("\n🧹 Cleaned up test files")

if __name__ == "__main__":
    main()