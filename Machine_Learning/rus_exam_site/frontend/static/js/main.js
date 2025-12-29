// Основной JavaScript для системы автоматической оценки экзаменов

document.addEventListener('DOMContentLoaded', function() {
    // Элементы DOM
    const uploadForm = document.getElementById('uploadForm');
    const uploadBtn = document.getElementById('uploadBtn');
    const progressContainer = document.getElementById('progressContainer');
    const progressBar = document.getElementById('progressBar');
    const progressText = document.getElementById('progressText');
    const resultContainer = document.getElementById('resultContainer');
    const errorContainer = document.getElementById('errorContainer');
    const resultText = document.getElementById('resultText');
    const errorText = document.getElementById('errorText');
    const downloadLink = document.getElementById('downloadLink');
    const csvFileInput = document.getElementById('csvFile');

    // Обработчик отправки формы
    uploadForm.addEventListener('submit', async function(e) {
        e.preventDefault();
        
        const file = csvFileInput.files[0];
        if (!file) {
            showError('Пожалуйста, выберите файл для загрузки');
            return;
        }

        // Валидация файла
        if (!file.name.toLowerCase().endsWith('.csv')) {
            showError('Пожалуйста, выберите файл в формате CSV');
            return;
        }

        if (file.size > 50 * 1024 * 1024) { // 50MB
            showError('Размер файла не должен превышать 50MB');
            return;
        }

        await uploadFile(file);
    });

    // Функция загрузки файла
    async function uploadFile(file) {
        const formData = new FormData();
        formData.append('file', file);

        // Показать прогресс
        showProgress();
        updateProgress(0, 'Подготовка к загрузке...');

        try {
            // Запускаем обработку
            const response = await fetch('/upload-csv/', {
                method: 'POST',
                body: formData
            });

            if (!response.ok) {
                const errorData = await response.json();
                throw new Error(errorData.detail || 'Ошибка сервера');
            }

            const result = await response.json();
            const taskId = result.task_id;
            
            // Отслеживаем прогресс
            await trackProgress(taskId);

        } catch (error) {
            console.error('Ошибка загрузки:', error);
            showError(error.message || 'Произошла ошибка при обработке файла');
        }
    }

    // Функция отслеживания прогресса
    async function trackProgress(taskId) {
        const interval = setInterval(async () => {
            try {
                const response = await fetch(`/status/${taskId}`);
                if (!response.ok) {
                    throw new Error('Ошибка получения статуса');
                }
                
                const status = await response.json();
                updateProgress(status.progress || 0, status.message || 'Обработка...');
                
                if (status.status === 'completed') {
                    clearInterval(interval);
                    showResult(status);
                } else if (status.status === 'error') {
                    clearInterval(interval);
                    showError(status.message || 'Ошибка обработки');
                }
            } catch (error) {
                clearInterval(interval);
                showError('Ошибка отслеживания прогресса');
            }
        }, 2000); // Проверяем каждые 2 секунды
    }

    // Функция показа прогресса
    function showProgress() {
        hideAllContainers();
        progressContainer.style.display = 'block';
        uploadBtn.disabled = true;
        uploadBtn.innerHTML = '<span class="spinner"></span> Обработка...';
    }

    // Функция обновления прогресса
    function updateProgress(percent, text) {
        progressBar.style.width = percent + '%';
        progressBar.setAttribute('aria-valuenow', percent);
        progressText.textContent = text;
    }

    // Функция показа результата
    function showResult(result) {
        hideAllContainers();
        resultContainer.style.display = 'block';
        
        resultText.innerHTML = `
            <strong>Файл успешно обработан!</strong><br>
            Обработано вопросов: <span class="badge bg-primary">${result.total_questions}</span><br>
            Имя файла: <code>${result.filename}</code>
        `;
        
        console.log('Настраиваем ссылку скачивания:', result.download_url);
        downloadLink.href = result.download_url;
        downloadLink.download = result.filename;
        
        // Добавляем обработчик клика на кнопку скачивания
        downloadLink.onclick = function(e) {
            console.log('Клик по кнопке скачивания:', this.href);
            // Принудительное скачивание
            window.open(this.href, '_blank');
        };
        
        resetForm();
        
        // Добавить анимацию
        resultContainer.classList.add('fade-in-up');
    }

    // Функция показа ошибки
    function showError(message) {
        hideAllContainers();
        errorContainer.style.display = 'block';
        
        // Разделяем сообщение на основное и детали
        const parts = message.split('\n\nПодробности:\n');
        const mainMessage = parts[0];
        const details = parts[1] || '';
        
        if (details) {
            errorText.innerHTML = `
                <div>${mainMessage}</div>
                <button class="btn btn-sm btn-outline-danger mt-2" onclick="toggleErrorDetails()">
                    <i class="fas fa-chevron-down" id="errorToggleIcon"></i> Показать подробности
                </button>
                <div id="errorDetails" style="display: none; margin-top: 10px;">
                    <pre style="background: #f8f9fa; padding: 10px; border-radius: 4px; font-size: 12px; max-height: 300px; overflow-y: auto;">${details}</pre>
                </div>
            `;
        } else {
            errorText.textContent = mainMessage;
        }
        
        resetForm();
        errorContainer.classList.add('fade-in-up');
    }
    
    // Функция переключения деталей ошибки
    window.toggleErrorDetails = function() {
        const details = document.getElementById('errorDetails');
        const icon = document.getElementById('errorToggleIcon');
        const button = icon.parentElement;
        
        if (details.style.display === 'none') {
            details.style.display = 'block';
            icon.className = 'fas fa-chevron-up';
            button.innerHTML = '<i class="fas fa-chevron-up"></i> Скрыть подробности';
        } else {
            details.style.display = 'none';
            icon.className = 'fas fa-chevron-down';
            button.innerHTML = '<i class="fas fa-chevron-down"></i> Показать подробности';
        }
    }

    // Функция скрытия всех контейнеров
    function hideAllContainers() {
        progressContainer.style.display = 'none';
        resultContainer.style.display = 'none';
        errorContainer.style.display = 'none';
        
        // Удалить классы анимации
        resultContainer.classList.remove('fade-in-up');
        errorContainer.classList.remove('fade-in-up');
    }

    // Функция сброса формы
    function resetForm() {
        uploadBtn.disabled = false;
        uploadBtn.innerHTML = '<i class="fas fa-cloud-upload-alt me-2"></i>Загрузить и обработать';
    }

    // Drag & Drop функциональность
    const uploadArea = document.querySelector('.card-body');
    
    ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
        uploadArea.addEventListener(eventName, preventDefaults, false);
    });

    function preventDefaults(e) {
        e.preventDefault();
        e.stopPropagation();
    }

    ['dragenter', 'dragover'].forEach(eventName => {
        uploadArea.addEventListener(eventName, highlight, false);
    });

    ['dragleave', 'drop'].forEach(eventName => {
        uploadArea.addEventListener(eventName, unhighlight, false);
    });

    function highlight(e) {
        uploadArea.classList.add('dragover');
    }

    function unhighlight(e) {
        uploadArea.classList.remove('dragover');
    }

    uploadArea.addEventListener('drop', handleDrop, false);

    function handleDrop(e) {
        const dt = e.dataTransfer;
        const files = dt.files;

        if (files.length > 0) {
            csvFileInput.files = files;
            
            // Показать имя файла
            const fileName = files[0].name;
            const fileLabel = document.querySelector('label[for="csvFile"]');
            fileLabel.textContent = `Выбран файл: ${fileName}`;
        }
    }

    // Обработчик изменения файла
    csvFileInput.addEventListener('change', function(e) {
        if (e.target.files.length > 0) {
            const fileName = e.target.files[0].name;
            const fileLabel = document.querySelector('label[for="csvFile"]');
            fileLabel.textContent = `Выбран файл: ${fileName}`;
        }
    });

    // Функция загрузки статистики
    async function loadStatistics() {
        try {
            const response = await fetch('/api/stats');
            if (response.ok) {
                const stats = await response.json();
                displayStatistics(stats);
            }
        } catch (error) {
            console.error('Ошибка загрузки статистики:', error);
        }
    }

    // Функция отображения статистики
    function displayStatistics(stats) {
        const statsContainer = document.getElementById('statsContainer');
        if (!statsContainer) return;

        statsContainer.innerHTML = `
            <div class="row">
                <div class="col-md-3 col-sm-6 mb-3">
                    <div class="stats-card">
                        <div class="stats-number">${stats.total_exams || 0}</div>
                        <div class="stats-label">Экзаменов</div>
                    </div>
                </div>
                <div class="col-md-3 col-sm-6 mb-3">
                    <div class="stats-card">
                        <div class="stats-number">${stats.total_questions || 0}</div>
                        <div class="stats-label">Вопросов</div>
                    </div>
                </div>
                <div class="col-md-3 col-sm-6 mb-3">
                    <div class="stats-card">
                        <div class="stats-number">${stats.total_answers || 0}</div>
                        <div class="stats-label">Ответов</div>
                    </div>
                </div>
                <div class="col-md-3 col-sm-6 mb-3">
                    <div class="stats-card">
                        <div class="stats-number">${calculateAccuracy(stats)}%</div>
                        <div class="stats-label">Точность</div>
                    </div>
                </div>
            </div>
        `;
    }

    // Функция расчета точности
    function calculateAccuracy(stats) {
        if (!stats.score_statistics || stats.score_statistics.length === 0) {
            return 0;
        }
        
        const avgError = stats.score_statistics.reduce((sum, item) => 
            sum + (item.avg_error || 0), 0) / stats.score_statistics.length;
        
        return Math.max(0, Math.round((1 - avgError / 2) * 100));
    }

    // Проверка состояния системы
    async function checkSystemHealth() {
        try {
            const response = await fetch('/health');
            if (response.ok) {
                const health = await response.json();
                updateHealthStatus(health);
            }
        } catch (error) {
            console.error('Ошибка проверки состояния системы:', error);
        }
    }

    // Обновление статуса системы
    function updateHealthStatus(health) {
        const statusIndicator = document.getElementById('systemStatus');
        if (!statusIndicator) return;

        const isHealthy = health.status === 'healthy' && 
                         health.ml_model_loaded && 
                         health.database_connected;

        statusIndicator.className = isHealthy ? 
            'badge bg-success' : 'badge bg-warning';
        statusIndicator.textContent = isHealthy ? 
            'Система работает' : 'Проблемы с системой';
    }

    // Плавная прокрутка для якорных ссылок
    document.querySelectorAll('a[href^="#"]').forEach(anchor => {
        anchor.addEventListener('click', function (e) {
            e.preventDefault();
            const target = document.querySelector(this.getAttribute('href'));
            if (target) {
                target.scrollIntoView({
                    behavior: 'smooth',
                    block: 'start'
                });
            }
        });
    });

    // Инициализация при загрузке страницы
    loadStatistics();
    checkSystemHealth();
    
    // Периодическое обновление статистики
    setInterval(loadStatistics, 30000); // каждые 30 секунд
    setInterval(checkSystemHealth, 60000); // каждую минуту
});

// Утилитарные функции
function formatFileSize(bytes) {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
}

function formatDate(dateString) {
    const date = new Date(dateString);
    return date.toLocaleDateString('ru-RU', {
        year: 'numeric',
        month: 'long',
        day: 'numeric',
        hour: '2-digit',
        minute: '2-digit'
    });
}

// Обработка ошибок глобально
window.addEventListener('error', function(e) {
    console.error('Глобальная ошибка:', e.error);
});

window.addEventListener('unhandledrejection', function(e) {
    console.error('Необработанное отклонение промиса:', e.reason);
});