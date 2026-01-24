# 🚀 Инструкция по развертыванию Heart Attack API на сервере

## Шаги развертывания:

### 1. Подготовка сервера

```bash
# Создание директории для приложения
sudo mkdir -p /var/www/heart_attack_api

# Создание виртуального окружения
sudo python3.9 -m venv /opt/heart_attack_venv39

# Активация окружения
source /opt/heart_attack_venv39/bin/activate

# Установка зависимостей
pip install fastapi uvicorn pandas numpy catboost scikit-learn pydantic python-multipart jinja2 aiofiles joblib requests
```

### 2. Загрузка файлов приложения

```bash
# Скопировать все файлы проекта в /var/www/heart_attack_api/
# Структура должна быть:
# /var/www/heart_attack_api/
# ├── app/
# ├── templates/
# ├── static/
# ├── models/
# └── data/
```

### 3. Настройка systemd сервиса

```bash
# Копирование файла сервиса
sudo cp heart_attack_api.service /etc/systemd/system/

# Перезагрузка systemd
sudo systemctl daemon-reload

# Включение автозапуска
sudo systemctl enable heart_attack_api.service

# Запуск сервиса
sudo systemctl start heart_attack_api.service

# Проверка статуса
sudo systemctl status heart_attack_api.service
```

### 4. Настройка nginx

```bash
# Копирование конфигурации nginx
sudo cp heart_attack_api.conf /etc/nginx/conf.d/

# Проверка конфигурации nginx
sudo nginx -t

# Перезагрузка nginx
sudo systemctl reload nginx
```

### 5. Проверка работы

```bash
# Проверка доступности API
curl http://localhost:8001/health

# Проверка через внешний адрес
curl http://vds.spb.su:8001/health
```

## 🔧 Управление сервисом:

```bash
# Остановка
sudo systemctl stop heart_attack_api.service

# Запуск
sudo systemctl start heart_attack_api.service

# Перезапуск
sudo systemctl restart heart_attack_api.service

# Просмотр логов
sudo journalctl -u heart_attack_api.service -f
```

## 🛠️ Устранение неполадок:

1. **Проверить порт 8001 свободен:**
   ```bash
   sudo netstat -tlnp | grep :8001
   ```

2. **Проверить логи сервиса:**
   ```bash
   sudo journalctl -u heart_attack_api.service --no-pager
   ```

3. **Проверить права доступа:**
   ```bash
   sudo chown -R root:root /var/www/heart_attack_api
   sudo chmod -R 755 /var/www/heart_attack_api
   ```