import streamlit as st
import numpy as np
import joblib
from tensorflow.keras.models import load_model

# Cargar recursos
model = load_model("modeloLSTM.keras")
scaler = joblib.load("my_scaler.pkl")

# Interfaz
st.title("🔮 Predictor de Series Temporales")
horas = st.slider("Selecciona horas a predecir:", 1, 48, 24)

# Mock data (¡Reemplaza con tus datos reales!)
ultimos_datos = np.random.rand(24).reshape(-1, 1)  # 24 horas históricas

if st.button("Generar predicción"):
    # Preprocesamiento
    datos_escalados = scaler.transform(ultimos_datos)
    entrada = datos_escalados.reshape(1, 24, 1)
    
    # Predicción
    prediccion = model.predict(entrada)
    prediccion_descalada = scaler.inverse_transform(prediccion)
    
    # Resultado
    st.line_chart({
        "Histórico": ultimos_datos.flatten(),
        "Predicción": prediccion_descalada.flatten()
    })