# Skin-ai

Skin-ai - это приложение на Streamlit для классификации кожных заболеваний с использованием обученной нейросети на PyTorch. Архитектуру использвовал ResNet-50 и обучил с разморозкой весов.

## Функционал
- Загрузка изображения кожи через веб-интерфейс
- Обнаружение и классификация заболевания
- Вывод предсказанного класса и уровня уверенности модели

### Датасет для обучения модели был взят отсюда 
https://www.kaggle.com/datasets/pacificrm/skindiseasedataset
## Установка и запуск

### Локальный запуск
1. Клонируйте репозиторий:
    ```sh
    git clone https://github.com/username/Skin-ai.git
    cd Skin-ai
    ```
2. Установите зависимости:
    ```sh
    pip install -r requirements.txt
    ```
3. Запустите приложение:
    ```sh
    streamlit run app.py
    ```

### Запуск через Docker
1. Соберите Docker-образ:
    ```sh
    docker build -t skin-ai .
    ```
2. Запустите контейнер:
    ```sh
    docker run -p 8501:8501 skin-ai
    ```
3. Откройте в браузере:
    ```
    http://localhost:8501
    ```

## Структура проекта
```
Skin-ai/
│── .github/workflows/      # GitHub Actions для CI/CD
│── .streamlit/             # Конфигурация Streamlit
│── Dockerfile              # Файл для сборки Docker-образа
│── LICENSE                 # Лицензия проекта
│── README.md               # Документация проекта
│── app.py                  # Основной код приложения
│── labels.txt              # Файл с метками классов
│── requirements.txt        # Список зависимостей
│── skin_disease_model_jit.pt # JIT-запакованная модель PyTorch
```

My telegram @eralyf
