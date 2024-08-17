import streamlit as st
from PIL import Image
import cv2
import numpy as np
import imutils
from tensorflow.keras.models import load_model
import gdown

# Configuración de la página
st.set_page_config(page_title="Segmentación de Tumor con CNN", page_icon="🧠", layout="centered")

# CSS personalizado para estilos y centrar el contenido
st.markdown(
    """
    <style>
    body {
        background-color: #2C3E50;  /* Azul noche */
        color: white;
        font-family: 'Arial', sans-serif;
    }
    .content {
        max-width: 700px; /* Reduce el ancho máximo */
        margin: 0 auto;
        padding: 20px;
    }
    .title {
        font-size: 48px;
        font-weight: bold;
        text-align: center;
        margin-bottom: 20px;
    }
    .subtitle {
        font-size: 24px;
        font-weight: bold;
        margin-top: 40px;
    }
    .justify-text {
        text-align: justify;
        font-size: 18px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Contenido de la aplicación centrado y con márgenes
st.markdown('<div class="content">', unsafe_allow_html=True)

# Títulos y secciones
st.markdown('<div class="title">Segmentación de Tumor con CNN</div>', unsafe_allow_html=True)
st.write("Proyecto para la clase de Reconocimeitno de Patrones.")
st.write("Dr. Harold Vazquez")
st.write("Integrantes: Diego Hernandez, Antonio Rocha, Ismael Mendoza")

st.markdown('<div class="subtitle">🎯 Objetivo del Proyecto</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="justify-text">El objetivo de esta aplicación es proporcionar una herramienta avanzada para la evaluación de segmentaciones de tumores en imágenes médicas. Utilizando técnicas de deep learning, la herramienta busca facilitar la tarea de los profesionales de la salud en la identificación y delimitación de áreas afectadas, mejorando así la precisión diagnóstica y el tratamiento subsecuente.</div>',
    unsafe_allow_html=True,
)

st.markdown('<div class="subtitle">📝 Justificación del Proyecto</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="justify-text">La segmentación precisa de tumores es crucial en la planificación y ejecución de tratamientos médicos, especialmente en oncología. Los métodos tradicionales, aunque efectivos, pueden ser propensos a errores humanos y varían significativamente en precisión. Esta aplicación ofrece una solución automatizada y consistente que no solo ahorra tiempo, sino que también minimiza la variabilidad entre los evaluadores, garantizando un estándar más elevado de atención médica.</div>',
    unsafe_allow_html=True,
)

st.markdown('<div class="subtitle">⚙️ Funcionamiento de la App</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="justify-text">La aplicación permite a los usuarios cargar imágenes médicas, que luego son procesadas por un modelo de segmentación basado en redes neuronales convolucionales (CNN). El sistema genera una segmentación automática del tumor, que puede ser visualizada y evaluada directamente en la plataforma. Además, la herramienta proporciona métricas clave que permiten a los médicos evaluar la precisión y efectividad de la segmentación generada.</div>',
    unsafe_allow_html=True,
)

st.markdown('<div class="subtitle">📤 Subir Imagen para Evaluación</div>', unsafe_allow_html=True)

# Función para procesar la imagen y predecir
def process_and_predict(image, model, img_size=(224, 224), add_pixels=0):
    def process_single_image_crop(img, add_pixels):
        img_array = np.array(img)
        
        if len(img_array.shape) == 2 or img_array.shape[2] == 1:
            # Imagen en escala de grises
            gray = img_array
        else:
            # Imagen en color, convertir a escala de grises
            gray = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)
        
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        
        thresh = cv2.threshold(gray, 45, 255, cv2.THRESH_BINARY)[1]
        thresh = cv2.erode(thresh, None, iterations=2)
        thresh = cv2.dilate(thresh, None, iterations=2)
        
        cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        
        if len(cnts) == 0:
            raise ValueError("No contours found in the image.")
        
        c = max(cnts, key=cv2.contourArea)
        
        extLeft = tuple(c[c[:, :, 0].argmin()][0])
        extRight = tuple(c[c[:, :, 0].argmax()][0])
        extTop = tuple(c[c[:, :, 1].argmin()][0])
        extBot = tuple(c[c[:, :, 1].argmax()][0])
        
        cropped_img = img_array[extTop[1] - add_pixels:extBot[1] + add_pixels, extLeft[0] - add_pixels:extRight[0] + add_pixels].copy()
        
        return cropped_img
    
    def preprocess_img(img, img_size):
        img = cv2.resize(img, dsize=img_size, interpolation=cv2.INTER_CUBIC)
        return img
    
    # Recortar y preprocesar la imagen
    cropped_img = process_single_image_crop(img=image, add_pixels=add_pixels)
    prep_img = preprocess_img(img=cropped_img, img_size=img_size)

    # Hacer predicciones
    predictions = model.predict(np.expand_dims(prep_img, axis=0))  # Añadir una dimensión para el batch
    
    # Asumiendo que la predicción es una probabilidad y quieres mostrar la precisión
    accuracy = np.max(predictions)  # Asume que `predictions` devuelve probabilidades
    
    # Retornar la predicción redondeada y la precisión
    prediction = np.round(predictions[0][0])
    
    return prediction, accuracy

# Descargar el modelo desde Google Drive
# Descargar el modelo desde Google Drive si no existe localmente
model_path = 'BrainTumor.keras'
gdrive_url = 'https://drive.google.com/uc?id=1kCua8wmGm_wExdT3IzfXVUXuezsSTrz9'

if not os.path.exists(model_path):
    gdown.download(gdrive_url, model_path, quiet=False, fuzzy=True)

# Cargar el modelo
model = load_model(model_path)

# Subir la imagen del tumor
uploaded_file = st.file_uploader("Sube una imagen del tumor", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Imagen subida.', use_column_width=True)
    
    # Cargar el modelo
    model_path = output  # Ajusta la ruta del modelo según tu sistema de archivos
    model = load_model(model_path)
    
    # Procesar la imagen y obtener predicción
    prediction, accuracy = process_and_predict(image, model, img_size=(224, 224), add_pixels=0)
    
    # Mostrar la predicción
    st.write(f'Predicción: {"Tumor detectado" if prediction == 1 else "No se detectó tumor"}')
    

st.markdown('</div>', unsafe_allow_html=True)  # Cierre del div content
