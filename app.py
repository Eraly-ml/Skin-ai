import streamlit as st
import torch
from torchvision import transforms
from PIL import Image
import os

# Настройки страницы
st.set_page_config(
    page_title="Классификация кожных заболеваний",
    page_icon="🩺",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Заголовок
st.title("🩺 Классификация кожных заболеваний")

# Предупреждение
st.warning(
    "⚠️ **Важно!** Данное приложение использует искусственный интеллект для анализа изображений, "
    "но оно **не является медицинским инструментом**. Результаты предсказания могут быть неточными. "
    "Для точной диагностики обратитесь к врачу-специалисту."
)

st.markdown("Загрузите изображение кожи, чтобы получить предсказание.")

# Ввод путей к модели и labels.txt
model_path = st.text_input("Введите путь к модели", "/home/eraly/Skin-ai/skin_disease_model_jit.pt")
labels_path = st.text_input("Введите путь к файлу labels.txt", "/home/eraly/Skin-ai/labels.txt")

# Проверка наличия модели
if not os.path.exists(model_path):
    st.error(f"Модель не найдена: {model_path}")
    st.stop()

# Загрузка модели
model = torch.jit.load(model_path, map_location=torch.device('cpu'))
model.eval()

# Проверка наличия labels.txt
if not os.path.exists(labels_path):
    st.error("Файл меток (labels.txt) не найден.")
    st.stop()

with open(labels_path, "r") as f:
    labels = [line.strip() for line in f.readlines()]

# Предобработка изображений
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

# Загрузка изображения
uploaded_file = st.file_uploader("Выберите изображение", type=["jpg", "jpeg", "png"], key="file_uploader")

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")  # Принудительно переводим в RGB
    st.image(image, caption="Загруженное изображение", use_container_width=True)

    # Отображение прогресса загрузки
    progress_bar = st.progress(0)
    for i in range(100):
        progress_bar.progress(i + 1)

    # Предсказание
    try:
        with st.spinner("Модель обрабатывает изображение..."):
            image_tensor = preprocess(image).unsqueeze(0)
            with torch.no_grad():
                output = model(image_tensor)

            scores = torch.nn.functional.softmax(output[0], dim=0)
            max_score_index = torch.argmax(scores).item()
            predicted_class = labels[max_score_index]
            confidence = scores[max_score_index].item()

        # Вывод результата
        st.markdown(f"""
            ### 🏥 Результат анализа:
            - **Результат:** `{predicted_class}`
            - **Вероятность результата:** `{confidence:.2f}`
        """)
    except Exception as e:
        st.error(f"Ошибка при классификации: {e}")

else:
    st.info("Пожалуйста, загрузите изображение.")
