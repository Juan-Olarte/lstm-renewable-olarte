import streamlit as st
import numpy as np
import joblib
import pandas as pd
import keras
from keras.models import load_model
#import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
import requests
from io import StringIO
from datetime import datetime


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

# Titulo de pestaña
st.set_page_config(page_title='Predicción Energías Renovables', layout='wide', page_icon="⚡")
# Interfaz
st.title("🔮 Predictor de Radiación Solar")
# You can also use "with" notation:
with tab1:
    st.radio("Select one:", [1, 2])
horas_a_predecir = st.slider("Selecciona horas a predecir:", 1, 48, 24)

# Cargar los datos históricos desde el archivo CSV
df = pd.read_csv("renewable_power_dataset_preprocesado.csv")  # Update path
ultimos_datos = df['ALLSKY_SFC_SW_DWN'].tail(24).values.reshape(-1, 1)  # Get last 24 hours

if st.button("Generar predicción"):
    # Preprocesamiento
    datos_escalados = scaler.transform(ultimos_datos)
    entrada = datos_escalados.reshape(1, 24, 1)  # Reshape for LSTM input

    # Predicción para múltiples horas
    predicciones = []
    for _ in range(horas_a_predecir):
        prediccion = model.predict(entrada)
        predicciones.append(prediccion[0, 0])  # Get the prediction value

        # Update input for next prediction (shift and add the new prediction)
        datos_escalados = np.roll(datos_escalados, -1)  # Shift data one step back
        datos_escalados[-1, 0] = prediccion[0, 0]  # Add the new prediction
        entrada = datos_escalados.reshape(1, 24, 1)

    # Desescalar las predicciones
    predicciones_descaladas = scaler.inverse_transform(np.array(predicciones).reshape(-1, 1))

    # Resultado
    st.line_chart({
        "Histórico": ultimos_datos.flatten(),
        "Predicción": predicciones_descaladas.flatten()
    })