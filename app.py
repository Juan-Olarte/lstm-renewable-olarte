import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
import pandas as pd

# 1) Carga del modelo
@st.cache(allow_output_mutation=True)
def load_lstm_model(path="modeloLSTM.h5"):
    return load_model(path)

model = load_lstm_model()

st.title("Predicción LSTM de series de tiempo")

# 2) Input del usuario
n_horas = st.number_input(
    "¿Cuántas horas hacia adelante quieres predecir?", 
    min_value=1, max_value=168, value=24, step=1
)

# 3) Preparar la secuencia de entrada para el modelo
# Aquí asumo que tienes guardada tu última ventana de datos en 'last_window.csv'
# que contiene exactamente el número de pasos de tiempo (timesteps) que tu modelo espera.
@st.cache
def load_last_window(path="last_window.csv"):
    return pd.read_csv(path).values  # shape (timesteps, features)

last_window = load_last_window()

# Genera predicción iterativa
def forecast(model, window, n_steps):
    history = window.copy()
    preds = []
    for _ in range(n_steps):
        # adapta las dimensiones: (1, timesteps, features)
        input_x = history.reshape((1, *history.shape))
        yhat = model.predict(input_x, verbose=0)
        # asumo que yhat shape = (1, features) o (1,1)
        preds.append(yhat.flatten())
        # avanza la ventana: elimina el primer paso, añade la predicción al final
        history = np.vstack((history[1:], yhat))
    return np.array(preds)

if st.button("Predecir"):
    with st.spinner("Generando predicción…"):
        preds = forecast(model, last_window, n_horas)
        # Si la salida es un solo valor (feature=1), aplanamos:
        if preds.shape[1] == 1:
            preds = preds.flatten()
        df_preds = pd.DataFrame({
            f"H+{i+1}": [v] for i, v in enumerate(preds)
        }).T.rename(columns={0: "Predicción"})
        st.line_chart(df_preds)

        st.dataframe(df_preds)
