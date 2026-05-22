# -*- coding: utf-8 -*-
"""
Подготовка данных для модели оценки водительского риска
"""

import pandas as pd
import psycopg2

# Параметры подключения к БД
db_config = {
    'user': 'praktikum_student',
    'pwd': 'Sdf4$2;d-d30pp',
    'host': 'rc1b-wcoijxj3yxfsf3fs.mdb.yandexcloud.net',
    'port': 6432,
    'db': 'data-science-vehicle-db'
}

connection_string = f"host={db_config['host']} port={db_config['port']} dbname={db_config['db']} user={db_config['user']} password={db_config['pwd']} sslmode=require"

def run_query(sql_query):
    """Выполняет SQL-запрос и возвращает результат в виде DataFrame."""
    with psycopg2.connect(connection_string) as conn:
        df = pd.read_sql(sql_query, conn)
    return df

# SQL-запрос для подготовки данных
query = """
SELECT 
    -- Целевая переменная
    p.at_fault,
    
    -- Факторы из таблицы parties
    p.party_type,
    p.party_sobriety,
    p.party_drug_physical,
    p.party_age,
    p.party_sex,
    p.cellphone_in_use,
    
    -- Факторы из таблицы collisions
    c.collision_date,
    c.collision_time,
    c.weather_1,
    c.road_surface,
    c.lighting,
    c.intersection,
    c.location_type,
    c.primary_collision_factor,
    c.type_of_collision,
    c.party_count,
    
    -- Факторы из таблицы vehicles
    v.vehicle_type,
    v.vehicle_transmission,
    v.vehicle_age
    
FROM collisions c
INNER JOIN parties p ON c.case_id = p.case_id
LEFT JOIN vehicles v ON c.case_id = v.case_id AND p.party_number = v.party_number

WHERE 
    -- Только машины (car)
    p.party_type = '1'
    
    -- Исключаем царапины (scratch)
    AND c.collision_damage != '0'
    
    -- Только данные за 2012 год
    AND EXTRACT(YEAR FROM c.collision_date) = 2012
"""

print("Загрузка данных...")
df = run_query(query)

print(f"\nЗагружено {len(df)} записей")
print(f"\nРазмерность данных: {df.shape}")

# Сохраняем данные
df.to_csv('accident_data_2012.csv', index=False)
print("\nДанные сохранены в файл 'accident_data_2012.csv'")

# Первичный анализ
print("\n=== ПЕРВИЧНЫЙ АНАЛИЗ ===")
print("\nИнформация о данных:")
print(df.info())

print("\n\nРаспределение целевой переменной (at_fault):")
print(df['at_fault'].value_counts())

print("\n\nПропущенные значения:")
print(df.isnull().sum())

print("\n\nОписательная статистика:")
print(df.describe())
