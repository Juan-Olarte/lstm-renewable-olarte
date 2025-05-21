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
st.set_page_config(page_title='Predicción Energías Renovables', layout='centered', page_icon="⚡")
# Interfaz
# Cabecera mejorada # MOD
st.markdown(
    "<h1 style='text-align: center; color: #0078D4;'>🔋 Predicción de Energías Renovables con IA</h1>",
    unsafe_allow_html=True
)
st.markdown("<hr style='border:1px solid #ddd;'>", unsafe_allow_html=True)
# You can also use "with" notation:
# Insert containers separated into tabs:
tab1, tab2, tab3 = st.tabs(["PREDICCIONES", "SOBRE NOSOTROS", "AYUDA Y TUTORIALES"])

#--------------------------------------------------------
#  PESTAÑA 1 -- PREDICCIONES
#--------------------------------------------------------
with tab1:
    st.info("🔧 Asegúrate de cargar un archivo válido o seleccionar una localización para comenzar.")
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
        horas_a_predecir = st.number_input(
            "Introduce el número de horas a predecir:",
            min_value=1,
            step=1,
            value=24
        )

        if horas_a_predecir > 48:
            st.warning("⚠️ El modelo solo permite predecir hasta 48 horas. Por favor, ingresa un número menor o igual a 48.")
        elif horas_a_predecir <= 0:
            st.warning("⚠️ Por favor, introduce un número positivo mayor a 0.")


        if st.button("Generar predicción"):
            # Escalado y reshape
            datos_escalados = scaler.transform(ultimos_datos)
            entrada = datos_escalados.reshape(1, 24, 1)

            # Generar predicciones
            predicciones = []
            for _ in range(horas_a_predecir):
                prediccion = model.predict(entrada)
                predicciones.append(prediccion[0, 0])

                datos_escalados = np.roll(datos_escalados, -1)
                datos_escalados[-1, 0] = prediccion[0, 0]
                entrada = datos_escalados.reshape(1, 24, 1)

            # Desescalar predicciones
            predicciones_descaladas = scaler.inverse_transform(np.array(predicciones).reshape(-1, 1))

            # Crear DataFrame para graficar
            total_puntos = 24 + horas_a_predecir
            serie_completa = [np.nan] * total_puntos
            historico = ultimos_datos.flatten().tolist()
            prediccion = predicciones_descaladas.flatten().tolist()

            # Asignar valores históricos y predichos
            for i in range(24):
                serie_completa[i] = historico[i]
            for i in range(horas_a_predecir):
                serie_completa[24 + i] = prediccion[i]

            # Crear índice temporal (puede ser horas ficticias)
            index = pd.RangeIndex(start=0, stop=total_puntos, step=1)

            df_resultado = pd.DataFrame({
                "Valor": serie_completa,
                "Tipo": ["Histórico"] * 24 + ["Predicción"] * horas_a_predecir
            }, index=index)

            # Mostrar gráfica
            st.subheader("Radiación solar predicha para el intervalo de tiempo seleccionado")
            st.line_chart(df_resultado.pivot(columns="Tipo", values="Valor"))

            # Calcular energía generada (Wh) con eficiencia del 27%
            eficiencia = 0.27
            area_m2 = 1
            perdidas = 0.8
            energia_wh = predicciones_descaladas * eficiencia * area_m2 * perdidas
            potencia_w = predicciones_descaladas * 1

            # Asegurarse que ambas listas sean 1D y tengan la misma longitud
            energia_wh = np.array(energia_wh).flatten()
            potencia_inst = np.array(potencia_w).flatten()

            if len(potencia_inst) == len(energia_wh):
                resultados_df = pd.DataFrame({
                    "Potencia instantánea (W)": potencia_inst
                })
                resultados2_df = pd.DataFrame({
                    "Energía generada (Wh)": energia_wh
                })

                st.subheader("Potencia instantánea estimada para un panel de 1m² (27% eficiencia)")
                st.line_chart(resultados_df)
                st.subheader("Energía estimada para un panel de 1m² (27% eficiencia)")
                st.line_chart(resultados2_df)
            else:
                st.error("Error: las dimensiones de radiación y energía no coinciden.")

            st.subheader("¿CUÁNTO DINERO AHORRARÍA UNA VIVIENDA?")
            promedio_energia = np.mean(energia_wh) * horas_a_predecir
            preciokwh = ['934.46','919.84','943.46','799.67','808.93']
            preciokwh = [float(p) for p in preciokwh]  # convierte a float
            ahorrokwh = [p * 0.001 * promedio_energia for p in preciokwh]

            ahorro = {
                'Ciudad': ['Cúcuta','Medellín','Bucaramanga','Cali','Bogotá'],
                'Precio KWh': preciokwh,
                'Dinero ahorrado por panel solar:': ahorrokwh
            }
            df_ahorro = pd.DataFrame(ahorro)
            st.table(df_ahorro)


    else:
        st.warning("🔍 Esperando que se carguen datos válidos con al menos 24 valores.")


#--------------------------------------------------------
#  PESTAÑA 2 -- SOBRE NOSOTROS
#--------------------------------------------------------

with tab2:
    st.subheader("Nuestro Equipo")
    st.image("imagenes/team.jpeg")
    st.markdown(
            "Juan David Olarte Pabón - Desarrollador Líder"
            "Deyson Iván Llanes Suárez - Desarrollador" 
            "Cristian David Alvarado Hernández - Desarrollador"
            "Angie Vanesa Montejo Montejo - Desarrolladora"
        )
    

#--------------------------------------------------------
#  PESTAÑA 3 -- AYUDA
#--------------------------------------------------------

with tab3:
    expand = st.expander("Como agregar localizaciones personalizadas", icon=":material/info:")
    expand2 = st.expander("Como hacer predicciones", icon=":material/info:")

    # You can also use "with" notation:
    with expand:
        st.subheader("CÓMO AGREGAR TUS PROPIAS LOCALIZACIONES")
        st.markdown("#### Dirígete a: https://power.larc.nasa.gov/data-access-viewer/")
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
        st.markdown("#### 1. Selecciona una localización predefinida o sube la tuya")
        st.image("imagenes/pred1.png")
        st.markdown("#### 2. Instroduce la cantidad de horas que deseas predecir")
        st.image("imagenes/pred2.png")
        st.markdown("#### 3. Haz click en el botón 'generar predicción'")
        st.image("imagenes/pred3.png")
        st.markdown("#### 4. ¡Observa tu predicción!")
        st.image("imagenes/pred4.png")
        st.image("imagenes/pred5.png")
        st.image("imagenes/pred6.png")