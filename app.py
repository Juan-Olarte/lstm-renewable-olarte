import streamlit as st
import numpy as np
import joblib
import keras
from keras.models import load_model

# Define y registra RMSE
@keras.saving.register_keras_serializable(name="RMSE")
def RMSE(y_true, y_pred):
    return keras.ops.sqrt(keras.ops.mean(keras.ops.square(y_pred - y_true), axis=-1))

# Carga el modelo con la función personalizada
model = load_model(
    "modeloLSTM.keras",
    custom_objects={"RMSE": RMSE}
)
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