#!/usr/bin/env python3
"""
Скрипт для запуска всех тестов приложения.
"""

import subprocess
import sys
import os


def run_tests():
    """Запуск всех тестов."""
    print("🧪 Запуск тестов Heart Attack API...")
    print("=" * 50)
    
    # Переходим в директорию проекта
    project_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(project_dir)
    
    try:
        # Запуск pytest
        result = subprocess.run([
            sys.executable, "-m", "pytest", 
            "tests/", 
            "-v", 
            "--tb=short",
            "--color=yes"
        ], check=True)
        
        print("\n✅ Все тесты прошли успешно!")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Тесты завершились с ошибкой. Код выхода: {e.returncode}")
        return False
    except FileNotFoundError:
        print("❌ pytest не найден. Установите: pip install pytest")
        return False


def run_coverage():
    """Запуск тестов с покрытием кода."""
    print("\n📊 Запуск тестов с анализом покрытия...")
    print("=" * 50)
    
    try:
        result = subprocess.run([
            sys.executable, "-m", "pytest", 
            "tests/", 
            "--cov=app",
            "--cov-report=html",
            "--cov-report=term-missing",
            "-v"
        ], check=True)
        
        print("\n📈 Отчет о покрытии создан в htmlcov/index.html")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Ошибка при создании отчета о покрытии: {e.returncode}")
        return False
    except FileNotFoundError:
        print("❌ pytest-cov не найден. Установите: pip install pytest-cov")
        return False


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Запуск тестов Heart Attack API")
    parser.add_argument("--coverage", action="store_true", 
                       help="Запустить тесты с анализом покрытия кода")
    
    args = parser.parse_args()
    
    if args.coverage:
        success = run_coverage()
    else:
        success = run_tests()
    
    sys.exit(0 if success else 1)