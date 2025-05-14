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
st.title("MODELO DE INTELIGENCIA ARTIFICIAL PARA PREDICCIÓN DE ENERGÍAS RENOVABLES")
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
        "Barrio El Contento - Cúcuta":"https://www.dropbox.com/scl/fi/pm33sppurztz0mmh6myjb/EL_CONTENTO_DATASET.csv?rlkey=mz4oo8y9v6svcvps2qo8yjjms&st=fh33wdj7&dl=0",
        "Barrio Aeropuerto - Cúcuta":"https://www.dropbox.com/scl/fi/n8d5w7kydi27548yvgpbw/EL_AEROPUERTO_DATASET.csv?rlkey=1uxesv5e36i7g1im5w6vo2fzr&st=w9kf6v63&dl=0",
        "Barrio Colsag - Cúcuta":"https://www.dropbox.com/scl/fi/2fsyd1fu8dnhehuknqmpd/COLSAG_DATASET.csv?rlkey=y8es8gtzlghw20zjqvxv6afzy&st=z9j1r50t&dl=0",
        "Patios Centro":"https://www.dropbox.com/scl/fi/j46wffrtcvsuscuvqlcui/PATIOS_CENTRO_DATASET.csv?rlkey=jvm3rd8kjlzjpx8cl8welpfiz&st=498wquac&dl=0",
        "El Zulia":"https://www.dropbox.com/scl/fi/7oafoa9gr8ckwlreh2sif/EL_ZULIA_DATASET.csv?rlkey=aq0d0y3hnn849jcypt664ycvz&st=xhq4kfha&dl=0",
        "Ureña":"https://www.dropbox.com/scl/fi/cav3zu16b8oaxlykvnslq/URENA_DATASET.csv?rlkey=jnl0frk7bfbpy60ebk8na61hy&st=upizhkt2&dl=0",
        "San Antonio del Táchira":"https://www.dropbox.com/scl/fi/43j8thmkin003rhd63mj8/SAN_ANTONIO_DATASET.csv?rlkey=patwxwmb7dzkz7xv2hli6vdej&st=jv433dhj&dl=0"
    }

    df = None

    #carga
    if location != "Seleccionar...":
        with st.spinner(f'Cargando datos de {location}...'):
            df = load_data(urls.get(location))
            if df is not None:
                st.success(f"✅ Datos cargados: {location}")
                st.session_state['data_source'] = location

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
            df = None  # Asegurar que no se use un df inválido

    # PREDICCIONES
    
    # Obtener últimos 24 valores
    if df is not None and 'ALLSKY_SFC_SW_DWN' in df.columns and len(df['ALLSKY_SFC_SW_DWN'].dropna()) >= 24:
        ultimos_datos = df['ALLSKY_SFC_SW_DWN'].tail(24).values.reshape(-1, 1)
        st.subheader("Seleccione el tiempo de predicción")
        horas_a_predecir = st.slider("Selecciona horas a predecir:", 1, 48, 24)

        # ----- CSS & Theme -----
        st.set_page_config(page_title="Predicción Climática IA", layout="wide")
        st.markdown(
            """
            <style>
            .stButton>button { background-color: #1f77b4; color: white; border-radius: 8px; }
            .css-1v3fvcr { text-align: center; }
            </style>
            """, unsafe_allow_html=True)

        # ----- Carga de Modelo y Scaler -----
        @st.cache(allow_output_mutation=True)
        def cargar_modelo():
            model = load_model("model_clima.h5")
            scaler = MinMaxScaler()
            # Cargar parámetros de scaler (suponiendo guardados en un archivo)
            scaler.min_, scaler.scale_ = np.load("scaler_params.npy", allow_pickle=True)
            return model, scaler

        model, scaler = cargar_modelo()

        # ----- Sidebar -----
        horas_a_predecir = st.sidebar.slider("Horas a predecir", min_value=1, max_value=48, value=24)
        ciudades = ["Cúcuta","Medellín","Bucaramanga","Cali","Bogotá"]
        ciudad = st.sidebar.selectbox("Ciudad", ciudades)
        precios_kwh = {
            "Cúcuta": 934.46,
            "Medellín": 919.84,
            "Bucaramanga": 943.46,
            "Cali": 799.67,
            "Bogotá": 808.93
        }

        # ----- Simulación de datos históricos -----
        # En tu caso, sustituir con la obtención real de ultimos_datos
        ultimos_datos = np.random.rand(24, 1)

        # ----- Función de Predicción -----
        def generar_prediccion(ultimos_datos, horas, model, scaler):
            datos_escalados = scaler.transform(ultimos_datos)
            entrada = datos_escalados.reshape(1, 24, 1)
            pred = []
            for i in range(horas):
                p = model.predict(entrada)[0, 0]
                pred.append(p)
                datos_escalados = np.roll(datos_escalados, -1)
                datos_escalados[-1, 0] = p
                entrada = datos_escalados.reshape(1, 24, 1)
            pred = np.array(pred).reshape(-1,1)
            return scaler.inverse_transform(pred).flatten()

        # ----- Botón y visualización -----
        if st.button("🌞 Generar predicción"):
            with st.spinner("Generando predicciones…"):
                progress = st.progress(0)
                predicciones = []
                # Escalado inicial\ n        datos_escalados = scaler.transform(ultimos_datos)
                entrada = datos_escalados.reshape(1,24,1)
                for i in range(horas_a_predecir):
                    p = model.predict(entrada)[0,0]
                    predicciones.append(p)
                    datos_escalados = np.roll(datos_escalados, -1)
                    datos_escalados[-1,0] = p
                    entrada = datos_escalados.reshape(1,24,1)
                    progress.progress((i+1)/horas_a_predecir)
                pred_des = scaler.inverse_transform(np.array(predicciones).reshape(-1,1)).flatten()
            st.success("¡Predicción completada!")

            # Datos completos
            total = 24 + horas_a_predecir
            serie = [np.nan]*total
            for i,v in enumerate(ultimos_datos.flatten()): serie[i] = v
            for j,v in enumerate(pred_des): serie[24+j] = v
            df = pd.DataFrame({
                "Hora": list(range(total)),
                "Valor": serie,
                "Tipo": ["Histórico"]*24 + ["Predicción"]*horas_a_predecir
            })

            # Métricas
            energia_wh = pred_des * 0.27 * 1 * 0.8\ n    potencia_w = pred_des
            ahorro = energia_wh.sum() * precios_kwh[ciudad] * 0.001

            col1, col2, col3 = st.columns(3)
            col1.metric("Potencia Pico (W)", f"{potencia_w.max():.1f}")
            col2.metric("Energía Total (Wh)", f"{energia_wh.sum():.0f}")
            col3.metric("Ahorro Estimado ($)", f"{ahorro:.0f}")

            # Pestañas
            tab1, tab2, tab3 = st.tabs(["Radiación", "Potencia/Energía", "Ahorro Económico"])
            with tab1:
                base = alt.Chart(df).encode(
                    x=alt.X("Hora:Q", title="Hora"),
                    y=alt.Y("Valor:Q", title="Radiación"),
                    color="Tipo:N",
                    tooltip=["Hora","Valor","Tipo"]
                )
                st.altair_chart(base.mark_line().interactive(), use_container_width=True)

            with tab2:
                df_pot = pd.DataFrame({"Hora": range(horas_a_predecir), "Potencia": potencia_w})
                df_ener = pd.DataFrame({"Hora": range(horas_a_predecir), "Energía": energia_wh})
                st.line_chart(df_pot.set_index("Hora"), height=300)
                st.line_chart(df_ener.set_index("Hora"), height=300)

            with tab3:
                df_ahorro = pd.DataFrame({
                    "Ciudad": ciudades,
                    "Precio KWh": [precios_kwh[c] for c in ciudades],
                    "Ahorro por panel ($)": [energia_wh.sum()*0.001*precios_kwh[c] for c in ciudades]
                })
                st.table(df_ahorro)


    else:
        st.warning("🔍 Esperando que se carguen datos válidos con al menos 24 valores.")


#--------------------------------------------------------
#  PESTAÑA 2 -- SOBRE NOSOTROS
#--------------------------------------------------------

with tab2:
    st.write("PLACEHOLDER")

#--------------------------------------------------------
#  PESTAÑA 3 -- AYUDA
#--------------------------------------------------------

with tab3:
    expand = st.expander("Como agregar localizaciones personalizadas", icon=":material/info:")
    expand2 = st.expander("Como agregar localizaciones personalizadas", icon=":material/info:")

    # You can also use "with" notation:
    with expand:
        st.subheader("CÓMO AGREGAR TUS PROPIAS LOCALIZACIONES")
        st.markdown("#### Primero debemos dirigirnos al siguiente enlace")
        st.markdown("https://power.larc.nasa.gov/data-access-viewer/")
        st.markdown(
            "NASA POWER es una base de datos de variables climáticas gestionada por la NASA."
            " De ella podemos obtener los datos necesarios para realizar predicciones al rededor" \
            " del mundo."
        )
        st.image("imagenes/mainpage.png")
        st.markdown(
            "#### Una vez en la página de NASA POWER deberemos seguir estos sencillos pasos:"
        )
        st.markdown(
            "##### 1. Nos ubicamos en la parte izquierda de la pantalla y seleccionamos 'single point'"
        )
        st.image("imagenes/selector.jpeg", width=300)
        st.markdown(
            "##### 2. Seleccionamos datos de energías renovables en 'community'"
        )
        st.image("imagenes/community.jpeg")
        st.markdown(
            "##### 3. Seleccionamos mediciones por hora, 'hourly'"
        )
        st.image("imagenes/tiempo.jpeg")
        st.markdown(
            "##### 4. Escogemos el primer parámetro, ALL SKY SURFACE SHORTWAVE DOWNWARD RADIATION"
        )
        st.image("imagenes/parametros.jpeg")
        st.markdown(
            "##### 5. Ubicamos en el mapa el lugar del que queremos obtener los datos"
        )
        st.image("imagenes/mapa.jpeg")
        st.markdown(
            "##### 6. Escogemos el formato .csv y descargamos los datos con el botón 'Submit'"
        )
        st.image("imagenes/formato.jpeg")
        st.image("imagenes/descarga.jpeg")
        st.markdown(
            "##### 7. Subimos los datos descargados la página web usando el botón 'Broswe Files'"
        )
        st.image("imagenes/browse.png")

    with expand2:




