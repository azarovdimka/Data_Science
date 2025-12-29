"""
Обработчик базы данных для экзаменационных данных
"""

import asyncpg
import pandas as pd
from typing import List, Dict, Any, Optional
from datetime import datetime
import json

class ExamHandler:
    """Класс для работы с базой данных экзаменов"""
    
    def __init__(self, database_url: str = None):
        self.database_url = database_url or "postgresql://exam_user:exam_password@localhost:5432/exam_db"
        self.pool = None
    
    async def init_connection_pool(self):
        """Инициализация пула соединений"""
        try:
            self.pool = await asyncpg.create_pool(
                self.database_url,
                min_size=1,
                max_size=10,
                command_timeout=60
            )
            print("Пул соединений с БД создан")
        except Exception as e:
            print(f"Ошибка подключения к БД: {e}")
    
    async def close_connection_pool(self):
        """Закрытие пула соединений"""
        if self.pool:
            await self.pool.close()
    
    async def check_connection(self) -> bool:
        """Проверка соединения с БД"""
        try:
            if not self.pool:
                await self.init_connection_pool()
            
            async with self.pool.acquire() as connection:
                await connection.fetchval("SELECT 1")
            return True
        except Exception as e:
            print(f"Ошибка соединения с БД: {e}")
            return False
    
    async def create_tables(self):
        """Создание таблиц в базе данных"""
        create_tables_sql = """
        -- Таблица экзаменов
        CREATE TABLE IF NOT EXISTS exams (
            id SERIAL PRIMARY KEY,
            exam_id VARCHAR(50) UNIQUE NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            status VARCHAR(20) DEFAULT 'active'
        );
        
        -- Таблица вопросов
        CREATE TABLE IF NOT EXISTS questions (
            id SERIAL PRIMARY KEY,
            question_id VARCHAR(50) UNIQUE NOT NULL,
            exam_id VARCHAR(50) REFERENCES exams(exam_id),
            question_number INTEGER NOT NULL,
            question_text TEXT NOT NULL,
            image_url VARCHAR(500),
            max_score INTEGER NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        
        -- Таблица ответов
        CREATE TABLE IF NOT EXISTS answers (
            id SERIAL PRIMARY KEY,
            question_id VARCHAR(50) REFERENCES questions(question_id),
            transcription TEXT NOT NULL,
            audio_url VARCHAR(500),
            examiner_score INTEGER,
            predicted_score INTEGER,
            confidence_score FLOAT,
            features JSONB,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        
        -- Таблица пользователей (для будущего расширения)
        CREATE TABLE IF NOT EXISTS users (
            id SERIAL PRIMARY KEY,
            username VARCHAR(100) UNIQUE NOT NULL,
            email VARCHAR(255) UNIQUE NOT NULL,
            password_hash VARCHAR(255) NOT NULL,
            role VARCHAR(20) DEFAULT 'user',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        
        -- Таблица сессий обработки файлов
        CREATE TABLE IF NOT EXISTS processing_sessions (
            id SERIAL PRIMARY KEY,
            session_id VARCHAR(100) UNIQUE NOT NULL,
            filename VARCHAR(255) NOT NULL,
            total_questions INTEGER NOT NULL,
            processed_questions INTEGER DEFAULT 0,
            status VARCHAR(20) DEFAULT 'processing',
            results JSONB,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            completed_at TIMESTAMP
        );
        
        -- Индексы для оптимизации
        CREATE INDEX IF NOT EXISTS idx_questions_exam_id ON questions(exam_id);
        CREATE INDEX IF NOT EXISTS idx_answers_question_id ON answers(question_id);
        CREATE INDEX IF NOT EXISTS idx_processing_sessions_status ON processing_sessions(status);
        """
        
        try:
            async with self.pool.acquire() as connection:
                await connection.execute(create_tables_sql)
            print("Таблицы успешно созданы")
        except Exception as e:
            print(f"Ошибка создания таблиц: {e}")
    
    async def save_exam_data(self, df: pd.DataFrame) -> str:
        """Сохранение данных экзамена в БД"""
        try:
            session_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            async with self.pool.acquire() as connection:
                async with connection.transaction():
                    # Сохранение сессии обработки
                    await connection.execute("""
                        INSERT INTO processing_sessions (session_id, filename, total_questions, status)
                        VALUES ($1, $2, $3, $4)
                    """, session_id, "uploaded_file.csv", len(df), "completed")
                    
                    # Сохранение данных по экзаменам и вопросам
                    for _, row in df.iterrows():
                        exam_id = str(row['Id экзамена'])
                        question_id = str(row['Id вопроса'])
                        
                        # Вставка экзамена (если не существует)
                        await connection.execute("""
                            INSERT INTO exams (exam_id) VALUES ($1)
                            ON CONFLICT (exam_id) DO NOTHING
                        """, exam_id)
                        
                        # Определение максимального балла
                        question_num = int(row['№ вопроса'])
                        max_score = 1 if question_num in [1, 3] else 2
                        
                        # Вставка вопроса
                        await connection.execute("""
                            INSERT INTO questions (question_id, exam_id, question_number, question_text, image_url, max_score)
                            VALUES ($1, $2, $3, $4, $5, $6)
                            ON CONFLICT (question_id) DO UPDATE SET
                                question_text = EXCLUDED.question_text,
                                image_url = EXCLUDED.image_url
                        """, question_id, exam_id, question_num, 
                            row['Текст вопроса'], 
                            row.get('Картинка из вопроса', ''), 
                            max_score)
                        
                        # Вставка ответа
                        await connection.execute("""
                            INSERT INTO answers (question_id, transcription, audio_url, examiner_score, predicted_score)
                            VALUES ($1, $2, $3, $4, $5)
                            ON CONFLICT DO NOTHING
                        """, question_id, 
                            row['Транскрибация ответа'],
                            row.get('Ссылка на оригинальный файл записи', ''),
                            row.get('Оценка экзаменатора'),
                            row.get('Predicted_Score'))
            
            return session_id
            
        except Exception as e:
            print(f"Ошибка сохранения данных: {e}")
            raise
    
    async def get_statistics(self) -> Dict[str, Any]:
        """Получение статистики системы"""
        try:
            async with self.pool.acquire() as connection:
                # Общая статистика
                total_exams = await connection.fetchval("SELECT COUNT(*) FROM exams")
                total_questions = await connection.fetchval("SELECT COUNT(*) FROM questions")
                total_answers = await connection.fetchval("SELECT COUNT(*) FROM answers")
                
                # Статистика по оценкам
                score_stats = await connection.fetch("""
                    SELECT 
                        q.question_number,
                        COUNT(*) as total_answers,
                        AVG(a.examiner_score) as avg_examiner_score,
                        AVG(a.predicted_score) as avg_predicted_score,
                        AVG(ABS(a.examiner_score - a.predicted_score)) as avg_error
                    FROM answers a
                    JOIN questions q ON a.question_id = q.question_id
                    WHERE a.examiner_score IS NOT NULL AND a.predicted_score IS NOT NULL
                    GROUP BY q.question_number
                    ORDER BY q.question_number
                """)
                
                # Последние сессии обработки
                recent_sessions = await connection.fetch("""
                    SELECT session_id, filename, total_questions, status, created_at
                    FROM processing_sessions
                    ORDER BY created_at DESC
                    LIMIT 10
                """)
                
                return {
                    "total_exams": total_exams,
                    "total_questions": total_questions,
                    "total_answers": total_answers,
                    "score_statistics": [dict(row) for row in score_stats],
                    "recent_sessions": [dict(row) for row in recent_sessions]
                }
                
        except Exception as e:
            print(f"Ошибка получения статистики: {e}")
            return {}
    
    async def get_exam_data(self, exam_id: str) -> List[Dict[str, Any]]:
        """Получение данных конкретного экзамена"""
        try:
            async with self.pool.acquire() as connection:
                rows = await connection.fetch("""
                    SELECT 
                        e.exam_id,
                        q.question_id,
                        q.question_number,
                        q.question_text,
                        q.image_url,
                        q.max_score,
                        a.transcription,
                        a.audio_url,
                        a.examiner_score,
                        a.predicted_score,
                        a.confidence_score
                    FROM exams e
                    JOIN questions q ON e.exam_id = q.exam_id
                    LEFT JOIN answers a ON q.question_id = a.question_id
                    WHERE e.exam_id = $1
                    ORDER BY q.question_number
                """, exam_id)
                
                return [dict(row) for row in rows]
                
        except Exception as e:
            print(f"Ошибка получения данных экзамена: {e}")
            return []