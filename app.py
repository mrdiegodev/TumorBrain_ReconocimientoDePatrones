import streamlit as st
from PIL import Image

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

# Sección de la funcionalidad principal
st.markdown('<div class="subtitle">📤 Subir Imagen para Evaluación</div>', unsafe_allow_html=True)

# Subir la imagen del tumor
uploaded_file = st.file_uploader("Sube una imagen del tumor", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Imagen subida.', use_column_width=True)
    st.write("La imagen se procesaría aquí cuando el modelo esté disponible.")

st.markdown('</div>', unsafe_allow_html=True)  # Cierre del div content
