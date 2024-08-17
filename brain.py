import streamlit as st
from PIL import Image
import cv2
import numpy as np
import imutils
from tensorflow.keras.models import load_model
import gdown

# Descargar el archivo desde Google Drive
gdrive_url = 'https://drive.google.com/uc?id=1kCua8wmGm_wExdT3IzfXVUXuezsSTrz9'
output = 'BrainTumor.keras'
gdown.download(gdrive_url, output, quiet=True)

# Cargar el modelo
model_path = "BrainTumor.keras"
model = load_model(model_path)

# Configuración de estilo CSS
page_bg_css = """
<style>
[data-testid="stAppViewContainer"] {
    background-color: #F0F2F6;  /* Cambia este color al que prefieras */
}

body, [data-testid="stAppViewContainer"] {
    color: #000000;  /* Cambia este color al que prefieras */
}

[data-testid="stHeader"] {
    background-color: rgba(0, 0, 0, 0);  /* Esto oculta el fondo del header */
}

[data-testid="stSidebar"] {
    background-color: #F0F2F6;  /* Cambia el color del sidebar */
}

h1, h2, h3, h4, h5, h6 {
    color: #333333;  /* Cambia este color al que prefieras */
}

div.stButton > button {
    background-color: #0073e6; /* Color de fondo del botón */
    color: #FFFFFF; /* Color del texto */
    padding: 10px 20px; /* Espaciado interno del botón */
    border-radius: 8px; /* Bordes redondeados */
    border: 2px solid #005bb5; /* Borde del botón */
    font-size: 16px; /* Tamaño de la fuente */
    font-weight: bold; /* Texto en negrita */
    box-shadow: 2px 2px 5px rgba(0, 0, 0, 0.2); /* Sombra del botón */
    cursor: pointer; /* Cambiar el cursor al pasar sobre el botón */
}

div.stButton > button:hover {
    background-color: #005bb5; /* Color de fondo al pasar el ratón */
    border-color: #003366; /* Mantener el color del borde al pasar el ratón */
}

.justified-text {
    text-align: justify;
}
</style>
"""

# Insertar el CSS en la aplicación
st.markdown(page_bg_css, unsafe_allow_html=True)

# Título de la aplicación
st.title("Reconocimiento de Tumores Cerebrales")
st.write("Esta aplicación utiliza un modelo de aprendizaje profundo para identificar tumores en imágenes de MRI cerebrales.")

# Instrucciones para el usuario
st.markdown("""
<div class="justified-text">
### Instrucciones de uso:
1. **Sube una imagen de MRI cerebral** utilizando el botón de abajo.
2. **El modelo analizará la imagen** y determinará si hay un tumor presente o no.
3. **Revisa el resultado** que se mostrará debajo de la imagen subida.
</div>
""", unsafe_allow_html=True)

# Cargar y procesar la imagen
uploaded_file = st.file_uploader("Elige una imagen de MRI cerebral...", type="jpg")

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Imagen de MRI cargada.', use_column_width=True)
    st.write("")
    st.write("Clasificando...")

    # Preprocesamiento de la imagen
    img_array = np.array(image.convert('RGB'))
    img_resized = cv2.resize(img_array, (64, 64))  # Asegúrate de que el tamaño de la imagen coincida con el input del modelo
    img_expanded = np.expand_dims(img_resized, axis=0)
    img_normalized = img_expanded / 255.0

    # Realizar la predicción
    prediction = model.predict(img_normalized)
    class_idx = np.argmax(prediction, axis=1)

    if class_idx == 0:
        st.write("El modelo predice: **No hay tumor**")
    else:
        st.write("El modelo predice: **Tumor detectado**")

