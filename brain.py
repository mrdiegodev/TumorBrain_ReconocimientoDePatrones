import streamlit as st
from PIL import Image
import cv2
import numpy as np
import imutils
from tensorflow.keras.models import load_model
import gdown
import os

# Configuración de estilo CSS
page_bg_css = """
<style>
[data-testid="stAppViewContainer"] {
    background-color: #FFF9E3;
}

body, [data-testid="stAppViewContainer"] {
    color: #040423;
}

[data-testid="stHeader"] {
    background-color: rgba(0, 0, 0, 0);
}

[data-testid="stSidebar"] {
    background-color: #040423;
}

h1, h2, h3, h4, h5, h6 {
    color: #333333;
}

div.stButton > button {
    background-color: #FFB6C1;
    color: #FFFFFF;
    padding: 10px 20px;
    border-radius: 12px;
    border: 2px solid #FF69B4;
    font-size: 16px;
    font-weight: bold;
    box-shadow: 2px 2px 5px rgba(0, 0, 0, 0.2);
    cursor: pointer;
}
div.stButton > button:hover {
    background-color: #FFC0CB;
    border-color: #FF69B4;
}

.justified-text {
    text-align: justify;
}
</style>
"""

# Insertar el CSS en la aplicación
st.markdown(page_bg_css, unsafe_allow_html=True)

# Título de la aplicación
st.title("Reconocimiento de Patrones en Tumores Cerebrales")
st.write("Proyecto para la clase de Reconocimiento de Patrones.")
st.write("👨‍🏫 Dr. Harold Vazquez")
st.write("🧑‍🎓 Integrantes: Steven Newton, Enrique Soto, Diego Hernandez")

# Descripción del proyecto
st.markdown("""
<div class="justified-text">
### 🎯 Objetivo del Proyecto
El proyecto tiene como objetivo desarrollar un modelo de aprendizaje profundo para la detección de tumores cerebrales a partir de imágenes de resonancia magnética (MRI). El modelo está basado en redes neuronales y se enfoca en identificar la presencia de tumores en las imágenes, proporcionando una herramienta de diagnóstico que apoye a los profesionales de la salud en la detección temprana de cáncer cerebral.

### 🔎 ¿Qué es una red neuronal?
Una red neuronal es un modelo de aprendizaje profundo que simula el funcionamiento del cerebro humano para procesar datos y reconocer patrones. Estas redes son especialmente útiles en el procesamiento de imágenes médicas, ya que pueden aprender a identificar características relevantes en las imágenes que son indicativas de enfermedades como el cáncer.

### 💻 Uso de la Aplicación
La aplicación permite a los usuarios cargar imágenes de resonancia magnética y utilizar el modelo entrenado para predecir la presencia de tumores cerebrales. El modelo proporcionará un diagnóstico preliminar basado en los patrones detectados en las imágenes cargadas.
</div>
""", unsafe_allow_html=True)

# Descargar el modelo desde Google Drive
gdrive_url = 'https://drive.google.com/uc?id=1kCua8wmGm_wExdT3IzfXVUXuezsSTrz9'
output = 'BrainTumor.keras'
gdown.download(gdrive_url, output, quiet=False)

# Cargar el modelo
model_path = "BrainTumor.keras"
model = load_model(model_path)

# Función para preprocesar la imagen
def preprocess_image(image):
    image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    image = cv2.resize(image, (150, 150))
    image = image / 255.0
    image = np.expand_dims(image, axis=0)
    return image

# Función para hacer predicciones
def predict_tumor(image):
    processed_image = preprocess_image(image)
    prediction = model.predict(processed_image)
    return prediction

# Subir una imagen
uploaded_file = st.file_uploader("Carga una imagen de MRI", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Imagen cargada", use_column_width=True)

    if st.button("🧠 Detectar Tumor"):
        st.write("Procesando la imagen...")
        prediction = predict_tumor(image)
        if prediction > 0.5:
            st.write("**El modelo detecta la presencia de un tumor.**")
        else:
            st.write("**El modelo no detecta la presencia de un tumor.**")
else:
    st.write("Por favor, sube una imagen para analizar.")

