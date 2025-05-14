import streamlit as st
import numpy as np
import joblib
import pandas as pd
import keras
from keras.models import load_model
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
    "modeloLSTM.keras",
    custom_objects={"RMSE": RMSE}
)
scaler = joblib.load("my_scaler.pkl")

@st.cache_data
def load_data(url):
    """Carga solo la quinta columna desde una URL con caché para mejor rendimiento"""
    try:
        if "dl=0" in url:
            url = url.replace("dl=0", "raw=1")
        elif "dl=1" in url:
            url = url.replace("dl=1", "raw=1")
        elif "raw=1" not in url:
            url += "&raw=1"

        response = requests.get(url)
        response.raise_for_status()
        df = pd.read_csv(StringIO(response.text), usecols=[4])
        return df
    except Exception as e:
        st.error(f"Error al cargar datos: {str(e)}")
        return None

#--------------------------------------------------
# CONFIGURACIÓN DE PÁGINA
#--------------------------------------------------

st.set_page_config(
    page_title='Predicción Energías Renovables',
    layout='wide',
    page_icon="⚡"
)

# Cabecera mejorada # MOD
st.markdown(
    "<h1 style='text-align: center; color: #0078D4;'>🔋 Predicción de Energías Renovables con IA</h1>",
    unsafe_allow_html=True
)
st.markdown("<hr style='border:1px solid #ddd;'>", unsafe_allow_html=True)

# Pestañas
tab1, tab2, tab3 = st.tabs([
    "🔮 PREDICCIONES",
    "ℹ️ SOBRE NOSOTROS",
    "❓ AYUDA Y TUTORIALES"
])

#--------------------------------------------------------
#  PESTAÑA 1 -- PREDICCIONES
#--------------------------------------------------------
with tab1:
    st.info("🔧 Asegúrate de cargar un archivo válido o seleccionar una localización para comenzar.")  # MOD

    st.subheader("📍 Seleccione la localización que desea usar para la predicción")
    location = st.selectbox("📍 Elige una localización:", [  # MOD
        "Seleccionar...",
        "Barrio El Contento - Cúcuta",
        "Barrio Aeropuerto - Cúcuta",
        "Barrio Colsag - Cúcuta",
        "Patios Centro",
        "El Zulia",
        "Ureña",
        "San Antonio del Táchira"
    ])

    # URLs predefinidas
    urls = {
        "Seleccionar...": "https://www.dropbox.com/scl/fi/8rfkm0866t6n9toqgzdyp/renewable_power_dataset_preprocesado.csv?raw=1",
        "Barrio El Contento - Cúcuta": "https://www.dropbox.com/scl/fi/pm33sppurztz0mmh6myjb/EL_CONTENTO_DATASET.csv?raw=1",
        "Barrio Aeropuerto - Cúcuta": "https://www.dropbox.com/scl/fi/n8d5w7kydi27548yvgpbw/EL_AEROPUERTO_DATASET.csv?raw=1",
        "Barrio Colsag - Cúcuta": "https://www.dropbox.com/scl/fi/2fsyd1fu8dnhehuknqmpd/COLSAG_DATASET.csv?raw=1",
        "Patios Centro": "https://www.dropbox.com/scl/fi/j46wffrtcvsuscuvqlcui/PATIOS_CENTRO_DATASET.csv?raw=1",
        "El Zulia": "https://www.dropbox.com/scl/fi/7oafoa9gr8ckwlreh2sif/EL_ZULIA_DATASET.csv?raw=1",
        "Ureña": "https://www.dropbox.com/scl/fi/cav3zu16b8oaxlykvnslq/URENA_DATASET.csv?raw=1",
        "San Antonio del Táchira": "https://www.dropbox.com/scl/fi/43j8thmkin003rhd63mj8/SAN_ANTONIO_DATASET.csv?raw=1"
    }

    df = None

    # Contenedor para carga de datos # MOD
    with st.container():
        if location != "Seleccionar...":
            with st.spinner(f'Cargando datos de {location}...'):
                df = load_data(urls.get(location))
                if df is not None:
                    st.success(f"✅ Datos cargados: {location}")
                    st.session_state['data_source'] = location

        uploaded_file = st.file_uploader("📁 Usa la pestaña de ayuda para subir tus propios CSV", type=['csv'])  # MOD
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
                df = None

    # Predicciones
    if df is not None and 'ALLSKY_SFC_SW_DWN' in df.columns and len(df['ALLSKY_SFC_SW_DWN'].dropna()) >= 24:
        ultimos_datos = df['ALLSKY_SFC_SW_DWN'].tail(24).values.reshape(-1, 1)
        st.subheader("⏱️ Seleccione el tiempo de predicción")
        horas_a_predecir = st.slider("Horas a predecir:", 1, 48, 24)

        if st.button("🔮 Generar predicción"):  # MOD
            datos_escalados = scaler.transform(ultimos_datos)
            entrada = datos_escalados.reshape(1, 24, 1)
            predicciones = []
            for _ in range(horas_a_predecir):
                pred = model.predict(entrada)
                predicciones.append(pred[0, 0])
                datos_escalados = np.roll(datos_escalados, -1)
                datos_escalados[-1, 0] = pred[0, 0]
                entrada = datos_escalados.reshape(1, 24, 1)
            pred_descal = scaler.inverse_transform(np.array(predicciones).reshape(-1, 1)).flatten()
            total = 24 + horas_a_predecir
            serie = [np.nan]*total
            historico = ultimos_datos.flatten().tolist()
            serie[:24] = historico
            serie[24:] = pred_descal.tolist()
            df_resultado = pd.DataFrame({
                "Valor": serie,
                "Tipo": ["Histórico"]*24 + ["Predicción"]*horas_a_predecir
            })
            promedio_energia = np.mean(pred_descal) * horas_a_predecir
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Horas predicción", horas_a_predecir)
            with col2:
                st.metric("Promedio energía (Wh)", f"{promedio_energia:.2f}")
            with st.expander("📈 Ver gráfica de radiación"):
                st.subheader("🌤️ Radiación solar predicha")
                st.line_chart(df_resultado.pivot(columns="Tipo", values="Valor"))
            eficiencia, area_m2, perdidas = 0.27, 1, 0.8
            energia_wh = pred_descal * eficiencia * area_m2 * perdidas
            potencia_w = pred_descal * 1
            with st.expander("⚡ Potencia instantánea estimada"):
                st.line_chart(potencia_w)
            with st.expander("🔋 Energía estimada"):
                st.line_chart(energia_wh)
            preciokwh = [934.46, 919.84, 943.46, 799.67, 808.93]
            ciudades = ['Cúcuta','Medellín','Bucaramanga','Cali','Bogotá']
            ahorrokwh = [p * 0.001 * promedio_energia for p in preciokwh]
            df_ahorro = pd.DataFrame({
                "Ciudad": ciudades,
                "Precio KWh": preciokwh,
                "Ahorro por panel (COP)": ahorrokwh
            })
            st.subheader("💰 Ahorro económico estimado")
            st.dataframe(
                df_ahorro.style.format({
                    "Precio KWh": "${:,.2f}",
                    "Ahorro por panel (COP)": "${:,.2f}"
                })
            )
    else:
        st.warning("🔍 Esperando datos válidos con al menos 24 valores.")

#--------------------------------------------------------
#  PESTAÑA 2 -- SOBRE NOSOTROS
#--------------------------------------------------------
with tab2:
    st.write("PLACEHOLDER")

#--------------------------------------------------------
#  PESTAÑA 3 -- AYUDA Y TUTORIALES
#--------------------------------------------------------
with tab3:
    expand = st.expander("ℹ️ Cómo agregar localizaciones personalizadas", expanded=True)
    with expand:
        st.subheader("CÓMO AGREGAR TUS PROPIAS LOCALIZACIONES")
        st.markdown("#### Dirígete a: https://power.larc.nasa.gov/data-access-viewer/")
        st.markdown(
            "1. Selecciona 'single point' en la parte izquierda.  " \

            "2. Elige 'community' → 'hourly' → 'ALL SKY SURFACE SHORTWAVE DOWNWARD RADIATION'.  "\

            "3. Ubica el punto en el mapa y descarga el CSV.  "\

            "4. Sube el CSV usando el botón de 'Browse Files'."
        )
        st.image("imagenes/mainpage.png")
        # ... resto de imágenes y pasos ...
