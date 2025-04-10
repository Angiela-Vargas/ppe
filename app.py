import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import tempfile
import os
import requests
from io import BytesIO

# Función para cargar el modelo con cache
@st.cache_resource
def load_model():
    model_path = "best.torchscript"
    if not os.path.exists(model_path):
        st.error("El modelo best.torchscript no se encontró.")
        return None
    return YOLO(model_path)

model = load_model()
if model is None:
    st.stop()

st.title("🦺 Detector de EPP")
st.markdown("Detecta: **Botas, Guantes, Casco, Persona, Chaleco**")

icons = {
    "botas": "🥾",
    "guantes": "🧤",
    "casco": "⛑️",
    "persona": "🧍",
    "chaleco": "🦺"
}

# 🔽 Selector de entrada
st.subheader("📷 Carga de imagen")
input_option = st.radio("Selecciona una opción", ["Subir desde tu equipo", "Desde URL"])

image = None

if input_option == "Subir desde tu equipo":
    uploaded_file = st.file_uploader("📤 Sube una imagen", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")

elif input_option == "Desde URL":
    url = st.text_input("🔗 Ingresa la URL de la imagen")
    if url:
        try:
            response = requests.get(url)
            image = Image.open(BytesIO(response.content)).convert("RGB")
        except Exception as e:
            st.error(f"No se pudo cargar la imagen desde la URL. Error: {e}")

# Si hay imagen, hacer predicción
if image is not None:
    st.image(image, caption="Imagen cargada", use_column_width=True)

    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
        image.save(tmp.name)
        path = tmp.name

    try:
        with st.spinner("🔍 Procesando imagen..."):
            results = model.predict(path, imgsz=640)
        st.success("✅ Procesamiento completado")
    except Exception as e:
        st.error(f"Error al realizar la predicción: {e}")
        results = None

    if results:
        st.image(results[0].plot(), caption="Detecciones", use_column_width=True)
        labels_detected = [results[0].names[int(cls)] for cls in results[0].boxes.cls]
        labels_detected = list(set(labels_detected))
        st.markdown("### 🧾 Objetos detectados:")
        if labels_detected:
            for label in labels_detected:
                icon = icons.get(label.lower(), "✅")
                st.write(f"{icon} {label.capitalize()}")
        else:
            st.write("⚠️ No se detectaron objetos.")


