import streamlit as st
from app.assets.Modulo2Web import Visualizador

st.set_page_config(page_title="Interfaz M2 AgroIA", layout="wide", page_icon="📊")

visualizador = Visualizador()

st.markdown('# AgroIA: Asistente Inteligente para el cultivo de la papa en Cartago, Costa Rica')
st.markdown('#### AgroIA es un sistema inteligente en Python que apoya a agricultores en Costa Rica. '
                'Usa IA (ARR, CNN, RNN) para detectar enfermedades en hojas, predecir rendimientos y '
                'recomendar acciones (riego, fertilización, poda), promoviendo una agricultura más productiva y sostenible.')
visualizador.cargar_formulario()
