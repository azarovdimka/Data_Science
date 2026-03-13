import pandas as pd
import numpy as np
import random

np.random.seed(42)
random.seed(42)

# Списки для генерации данных
room_types = ['Стандарт', 'Люкс', 'Делюкс', 'Апартаменты', 'Студия', 'Семейный', 'Эконом', 'Бизнес']
bed_types = ['Двуспальная кровать', 'Две односпальные', 'Односпальная', 'King Size', 'Queen Size', '']
view_types = ['Вид на море', 'Вид на город', 'Вид на горы', 'Вид на бассейн', 'Без окон', '']
amenities_list = ['Wi-Fi', 'Кондиционер', 'Телевизор', 'Мини-бар', 'Сейф', 'Балкон', 'Ванна', 'Душ', 'Фен', 'Халаты']
operators = ['Booking.com', 'Expedia', 'Hotels.ru', 'Ostrovok', 'TravelLine', 'Яндекс.Путешествия']

def generate_room_data(n_samples=10000):
    data = []
    
    for i in range(n_samples):
        # Определяем, будет ли запись неполной (30% неполных)
        is_incomplete = np.random.random() < 0.3
        
        # Базовые поля
        room_name = random.choice(room_types) if np.random.random() > 0.1 else ''
        operator = random.choice(operators)
        
        # Площадь комнаты
        area = round(np.random.uniform(15, 80), 1) if np.random.random() > 0.15 else None
        
        # Тип кровати
        if is_incomplete and np.random.random() < 0.5:
            bed_type = ''
        else:
            bed_type = random.choice(bed_types)
        
        # Вид из окна
        if is_incomplete and np.random.random() < 0.4:
            view = ''
        else:
            view = random.choice(view_types)
        
        # Количество гостей
        max_guests = np.random.randint(1, 5) if np.random.random() > 0.2 else None
        
        # Удобства
        if is_incomplete:
            n_amenities = np.random.randint(0, 3)
        else:
            n_amenities = np.random.randint(3, 8)
        amenities = ', '.join(random.sample(amenities_list, n_amenities))
        
        # Описание
        desc_length = np.random.randint(10, 50) if not is_incomplete else np.random.randint(0, 15)
        description = ' '.join(['слово'] * desc_length) if desc_length > 0 else ''
        
        # Этаж
        floor = np.random.randint(1, 15) if np.random.random() > 0.3 else None
        
        # Количество комнат
        n_rooms = np.random.randint(1, 4) if np.random.random() > 0.25 else None
        
        # Цена
        price = round(np.random.uniform(2000, 25000), 2)
        
        # Рейтинг отеля
        rating = round(np.random.uniform(3.5, 5.0), 1) if np.random.random() > 0.1 else None
        
        # Количество фотографий
        n_photos = np.random.randint(0, 15)
        
        # Целевая переменная: 1 - неполные данные, 0 - полные
        target = 1 if is_incomplete else 0
        
        data.append({
            'id': i,
            'operator': operator,
            'room_name': room_name,
            'area': area,
            'bed_type': bed_type,
            'view': view,
            'max_guests': max_guests,
            'amenities': amenities,
            'description': description,
            'floor': floor,
            'n_rooms': n_rooms,
            'price': price,
            'rating': rating,
            'n_photos': n_photos,
            'is_incomplete': target
        })
    
    return pd.DataFrame(data)

# Генерация данных
df = generate_room_data(10000)

# Сохранение
df.to_csv('hotel_rooms_data.csv', index=False, encoding='utf-8-sig')
print(f"Данные сгенерированы: {len(df)} записей")
print(f"Неполных описаний: {df['is_incomplete'].sum()} ({df['is_incomplete'].mean()*100:.1f}%)")
print("\nПервые строки:")
print(df.head())
