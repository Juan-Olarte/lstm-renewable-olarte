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

#--------------------------------------------------
# DEFINICIONES
#--------------------------------------------------

@keras.saving.register_keras_serializable(name="RMSE")
def RMSE(y_true, y_pred):
    return keras.ops.sqrt(keras.ops.mean(keras.ops.square(y_pred - y_true), axis=-1))

# Carga el modelo con la función personalizada
model = load_model(
    "modeloLSTM.keras",  # Update path
    custom_objects={"RMSE": RMSE}
)
scaler = joblib.load("my_scaler.pkl")  # Update path

@st.cache_data
def load_data(url):
    """Carga solo la quinta columna desde una URL con caché para mejor rendimiento"""
    try:
        # Asegura que sea un enlace de descarga directa
        if "dl=0" in url:
            url = url.replace("dl=0", "raw=1")
        elif "dl=1" in url:
            url = url.replace("dl=1", "raw=1")
        elif "raw=1" not in url:
            url += "&raw=1"

        response = requests.get(url)
        response.raise_for_status()

        # Leer solo la 5ta columna (índice 4)
        df = pd.read_csv(StringIO(response.text), usecols=[4])

        return df
    except Exception as e:
        st.error(f"Error al cargar datos: {str(e)}")
        return None


# Titulo de pestaña
st.set_page_config(page_title='Predicción Energías Renovables', layout='wide', page_icon="⚡")
# Interfaz
st.header("MODELO DE INTELIGENCIA ARTIFICIAL PARA PREDICCIÓN DE ENERGÍAS RENOVABLES")
# You can also use "with" notation:
# Insert containers separated into tabs:
tab1, tab2, tab3 = st.tabs(["PREDICCIONES", "SOBRE NOSOTROS", "AYUDA Y TUTORIALES"])
tab1.write("this is tab 1")
tab2.write("this is tab 2")
tab3.write("this is tab 3")

#--------------------------------------------------------
#  PESTAÑA 1 -- PREDICCIONES
#--------------------------------------------------------
with tab1:
    st.subheader("Seleccione la localización que desea usar para la predicción")
    #Opciones predefinidas
    location = st.selectbox("Localizaciones predfinidas", [
        "Seleccionar...",
        "Barrio El Contento - Cúcuta",
        "Barrio Aeropuerto - Cúcuta",
        "Barrio Colsag - Cúcuta",
        "Patios Centro",
        "El Zulia",
        "Ureña",
        "San Antonio del Táchira"
    ])
    
    #urls
    urls = {
        "Seleccionar...":"https://www.dropbox.com/scl/fi/8rfkm0866t6n9toqgzdyp/renewable_power_dataset_preprocesado.csv?rlkey=g08nlgjt6y2dg9jm5iv7g3hlp&st=xvot6kwq&dl=0",
        "Barrio El Contento - Cúcuta":"https://www.dropbox.com/scl/fi/pm33sppurztz0mmh6myjb/EL_CONTENTO_DATASET.csv?rlkey=mz4oo8y9v6svcvps2qo8yjjms&st=f522sgts&dl=0",
        "Barrio Aeropuerto - Cúcuta":"https://www.dropbox.com/scl/fi/n8d5w7kydi27548yvgpbw/EL_AEROPUERTO_DATASET.csv?rlkey=1uxesv5e36i7g1im5w6vo2fzr&st=bvflxtdx&dl=0",
        "Barrio Colsag - Cúcuta":"https://www.dropbox.com/scl/fi/2fsyd1fu8dnhehuknqmpd/COLSAG_DATASET.csv?rlkey=y8es8gtzlghw20zjqvxv6afzy&st=1apoakdo&dl=0",
        "Patios Centro":"https://www.dropbox.com/scl/fi/j46wffrtcvsuscuvqlcui/PATIOS_CENTRO_DATASET.csv?rlkey=jvm3rd8kjlzjpx8cl8welpfiz&st=zqh3xh47&dl=0",
        "El Zulia":"https://www.dropbox.com/scl/fi/7oafoa9gr8ckwlreh2sif/EL_ZULIA_DATASET.csv?rlkey=aq0d0y3hnn849jcypt664ycvz&st=xooctz44&dl=0",
        "Ureña":"https://www.dropbox.com/scl/fi/cav3zu16b8oaxlykvnslq/URENA_DATASET.csv?rlkey=jnl0frk7bfbpy60ebk8na61hy&st=0bjr7bwt&dl=0",
        "San Antonio del Táchira":"https://www.dropbox.com/scl/fi/43j8thmkin003rhd63mj8/SAN_ANTONIO_DATASET.csv?rlkey=patwxwmb7dzkz7xv2hli6vdej&st=nsr2f61j&dl=0"
    }

    df = None

    #carga
    if location != "Seleccionar...":
        with st.spinner(f'Cargando datos de {location}...'):
            df = load_data(urls.get(location))
            if df is not None:
                st.success(f"✅ Datos cargados: {location}")
                st.session_state['data_source'] = location
    horas_a_predecir = st.slider("Selecciona horas a predecir:", 1, 48, 24)

    uploaded_file = st.file_uploader("Usa la pestaña de ayuda y tutoriales para subir tus propias bases de datos", type=['csv'])

    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            if 'ALLSKY_SFC_SW_DWN' not in df.columns:
                st.error("El archivo debe contener la columna 'ALLSKY_SFC_SW_DWN'")
            else:
                st.success("✅ Archivo cargado correctamente")
                st.session_state['data_source'] = "Archivo personalizado"
        except Exception as e:
            st.error(f"Error al leer el archivo: {str(e)}")

    # PREDICCIONES
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

#--------------------------------------------------------
#  PESTAÑA 2 -- SOBRE NOSOTROS
#--------------------------------------------------------

with tab2:
    st.write("PLACEHOLDER")

#--------------------------------------------------------
#  PESTAÑA 3 -- AYUDA
#--------------------------------------------------------

with tab3:
    st.write("PLACEHOLDER")


