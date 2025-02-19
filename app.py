import streamlit as st
import torch
from torchvision import transforms
from PIL import Image
import os

model_path = "skin_disease_model_jit.pt"
if not os.path.exists(model_path):
    st.error(f"Модель не найдена: {model_path}")
    st.stop()

model = torch.jit.load(model_path, map_location=torch.device('cpu'))
model.eval()


if not os.path.exists("labels.txt"):
    st.error("Файл меток (labels.txt) не найден.")
    st.stop()

with open("labels.txt", "r") as f:
    labels = [line.strip() for line in f.readlines()]

preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])


st.set_page_config(
    page_title="Классификация кожных заболеваний",
    page_icon="🩺",
    layout="centered",
    initial_sidebar_state="collapsed"
)


st.markdown("""
    <style>
        .stApp {
            background-color: #f0f2f6;
            padding: 20px;
            border-radius: 10px;
        }
        h1 {
            color: #2c3e50;
            text-align: center;
        }
        .stFileUploader > div > div {
            background-color: #ecf0f1;
            border-radius: 10px;
        }
        .css-1d391kg {
            background-color: #ffffff;
            border-radius: 10px;
            padding: 20px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        }
    </style>
""", unsafe_allow_html=True)


st.title("🩺 Классификация кожных заболеваний")
st.markdown("Загрузите изображение кожи, чтобы получить предсказание.")


uploaded_file = st.file_uploader("Выберите изображение", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Загруженное изображение", use_column_width=True)


st.title("🩺 Классификация кожных заболеваний")
st.markdown("Загрузите изображение кожи, чтобы получить предсказание.")

uploaded_file = st.file_uploader("Выберите изображение", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Загруженное изображение", use_column_width=True)

    progress_bar = st.progress(0)
    for i in range(100):
        progress_bar.progress(i + 1)

    try:
        
        image_tensor = preprocess(image).unsqueeze(0)
        with torch.no_grad():
            output = model(image_tensor)
        

        image_tensor = preprocess(image).unsqueeze(0)
        with torch.no_grad():
            output = model(image_tensor)

        scores = torch.nn.functional.softmax(output[0], dim=0)
        max_score_index = torch.argmax(scores).item()
        predicted_class = labels[max_score_index]
        confidence = scores[max_score_index].item()



        st.markdown(f"""
            ### Результат:
            - **Предсказанный класс:** {predicted_class}
            - **Достоверность:** {confidence:.2f}
        """)
    except Exception as e:
        st.error(f"Ошибка при классификации: {e}")
else:
    st.info("Пожалуйста, загрузите изображение.")
