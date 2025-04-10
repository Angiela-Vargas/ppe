import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import tempfile
import os

# Función para cargar el modelo con cache
@st.cache_resource
def load_model():
    model_path = "best.pt"
    if not os.path.exists(model_path):
        st.error("El modelo best.pt no se encontró. Asegúrate de que esté en la misma carpeta que app.py.")
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
uploaded_file = st.file_uploader("📤 Sube una imagen", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Imagen original", use_column_width=True)

    # Guarda la imagen temporalmente
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
        image.save(tmp.name)
        path = tmp.name

    # Realiza la predicción
    try:
        with st.spinner("Procesando imagen..."):
            results = model.predict(path, imgsz=640)
        st.success("¡Procesamiento completado!")
    except Exception as e:
        st.error(f"Error al realizar la predicción: {e}")
        results = None

    if results is not None:
        # Muestra la imagen con detecciones
        st.image(results[0].plot(), caption="Detecciones", use_column_width=True)

        # Extrae y muestra los nombres de los objetos detectados
        labels_detected = [results[0].names[int(cls)] for cls in results[0].boxes.cls]
        labels_detected = list(set(labels_detected))
        st.markdown("### 🧾 Objetos detectados:")
        if labels_detected:
            for label in labels_detected:
                icon = icons.get(label.lower(), "✅")
                st.write(f"{icon} {label.capitalize()}")
        else:
            st.write("⚠️ No se detectaron objetos.")

