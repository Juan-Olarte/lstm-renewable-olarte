import streamlit as st
import numpy as np
import joblib
import pandas as pd  # Import pandas for data handling
import keras
from keras.models import load_model

# Define y registra RMSE
@keras.saving.register_keras_serializable(name="RMSE")
def RMSE(y_true, y_pred):
    return keras.ops.sqrt(keras.ops.mean(keras.ops.square(y_pred - y_true), axis=-1))

# Carga el modelo con la función personalizada
model = load_model(
    "modeloLSTM.keras",  # Update path
    custom_objects={"RMSE": RMSE}
)
scaler = joblib.load("my_scaler.pkl")  # Update path

# Interfaz
st.title("🔮 Predictor de Radiación Solar")
horas = st.slider("Selecciona horas a predecir:", 1, 48, 24)

# Cargar los datos históricos desde el archivo CSV
df = pd.read_csv("renewable_power_dataset_preprocesado.csv")  # Update path
ultimos_datos = df['ALLSKY_SFC_SW_DWN'].tail(24).values.reshape(-1, 1)  # Get last 24 hours

if st.button("Generar predicción"):
    # Preprocesamiento
    datos_escalados = scaler.transform(ultimos_datos)
    entrada = datos_escalados.reshape(1, 24, horas)  # Reshape for LSTM input
    
    # Predicción
    prediccion = model.predict(entrada)
    prediccion_descalada = scaler.inverse_transform(prediccion)
    
    # Resultado
    st.line_chart({
        "Histórico": ultimos_datos.flatten(),
        "Predicción": prediccion_descalada.flatten()
    })