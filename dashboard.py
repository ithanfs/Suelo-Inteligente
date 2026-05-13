import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier

# 1. CONFIGURACIÓN DE PÁGINA Y MODO NEÓN
st.set_page_config(page_title="Suelo Inteligente", page_icon="🌐", layout="wide")

st.markdown("""
<style>
    .stApp { background-color: #090a0f; }
    h1, h2, h3 { color: #00f3ff !important; font-family: 'Arial', sans-serif; }
    div[data-testid="metric-container"] { background-color: #12151e; border: 1px solid #1f2940; padding: 15px; border-radius: 10px; box-shadow: 0 0 15px rgba(0, 243, 255, 0.1); }
    div[data-testid="metric-container"] > div { color: #b026ff; }
    hr { border-color: #b026ff; opacity: 0.3; }
</style>
""", unsafe_allow_html=True)

# 2. ENTRENAR EL "CEREBRO" CON LOS DATOS HISTÓRICOS (SE HACE UNA SOLA VEZ)
@st.cache_resource
def entrenar_cerebro():
    df_base = pd.read_csv('Crop_recommendation.csv')
    df_base['Total_Nutrientes'] = df_base['N'] + df_base['P'] + df_base['K']
    condiciones = [
        (df_base['Total_Nutrientes'] >= 150),
        (df_base['Total_Nutrientes'] >= 80) & (df_base['Total_Nutrientes'] < 150),
        (df_base['Total_Nutrientes'] < 80)
    ]
    df_base['Nivel_Fertilidad'] = np.select(condiciones, ['Alta', 'Media', 'Baja'], default='Sin_Clasificar')
    
    X = df_base[['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']]
    y = df_base['Nivel_Fertilidad']
    
    escalador = StandardScaler()
    X_escalado = escalador.fit_transform(X)
    
    modelo = KNeighborsClassifier(n_neighbors=5)
    modelo.fit(X_escalado, y)
    
    return modelo, escalador

modelo_knn, escalador = entrenar_cerebro()

# 3. INTERFAZ: EL SISTEMA DE CARGA DE DATOS PARA EL CAMPESINO
st.markdown("<h1>🌐 Suelo inteligente: Motor de Diagnóstico Agrícola</h1>", unsafe_allow_html=True)
st.markdown("Sube tu archivo de registro de parcelas para obtener un diagnóstico impulsado por la ciencia de datos.")
st.markdown("---")

st.sidebar.markdown("### 📂 Ingresa tus datos")
st.sidebar.markdown("El archivo CSV debe contener las columnas: `N, P, K, temperature, humidity, ph, rainfall`.")
archivo_campesino = st.sidebar.file_uploader("Arrastra tu archivo CSV aquí", type=["csv"])

# 4. LÓGICA DE PREDICCIÓN Y VISUALIZACIÓN DINÁMICA
if archivo_campesino is not None:
    # Si el campesino subió un archivo, lo leemos
    df_usuario = pd.read_csv(archivo_campesino)
    st.success("¡Datos cargados correctamente! Analizando huella del suelo...")
    
    # Predecimos usando el cerebro que ya entrenamos
    X_usuario = df_usuario[['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']]
    datos_escalados = escalador.transform(X_usuario)
    df_usuario['Diagnostico_Fertilidad'] = modelo_knn.predict(datos_escalados)
    
    # Mostramos los KPIs del campesino
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Parcelas Analizadas", f"{len(df_usuario):,}")
    col2.metric("pH Promedio", f"{df_usuario['ph'].mean():.2f}")
    col3.metric("Lluvia Promedio", f"{df_usuario['rainfall'].mean():.1f} mm")
    
    parcelas_altas = len(df_usuario[df_usuario['Diagnostico_Fertilidad'] == 'Alta'])
    col4.metric("Tierras Óptimas (Alta)", f"{parcelas_altas}")
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Mostramos las gráficas generadas con los datos del campesino
    c_izq, c_der = st.columns(2)
    with c_izq:
        st.markdown("### 🎯 Proporción de Calidad en tu Terreno")
        fig_pie = px.pie(df_usuario, names='Diagnostico_Fertilidad', hole=0.6, template='plotly_dark',
                         color='Diagnostico_Fertilidad',
                         color_discrete_map={'Alta':'#00ff9d', 'Media':'#00f3ff', 'Baja':'#b026ff'})
        fig_pie.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', margin=dict(t=30, b=0))
        st.plotly_chart(fig_pie, use_container_width=True)
        
    with c_der:
        st.markdown("### 📊 Composición Química Promedio")
        df_agrupado = df_usuario.groupby('Diagnostico_Fertilidad')[['N', 'P', 'K']].mean().reset_index()
        fig_bar = px.bar(df_agrupado, x='Diagnostico_Fertilidad', y=['N', 'P', 'K'], barmode='group', template='plotly_dark',
                         color_discrete_sequence=['#00f3ff', '#b026ff', '#00ff9d'])
        fig_bar.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', margin=dict(t=30, b=0))
        st.plotly_chart(fig_bar, use_container_width=True)

    # Mostramos la tabla final para que pueda descargarla
    st.markdown("### 📋 Tabla de Diagnóstico Detallada")
    st.dataframe(df_usuario, use_container_width=True)

else:
    # Si no ha subido nada, le damos instrucciones
    st.info("👈 Por favor, carga tu archivo CSV en el panel lateral para comenzar el análisis.")