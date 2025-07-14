import os

import streamlit as st
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.assets.Modulo2Web import Modulo2Web

st.set_page_config(page_title="Interfaz M2 AgroIA", layout="wide", page_icon="📊")

visualizador = Modulo2Web()

st.markdown('# AgroIA: Asistente Inteligente para el cultivo de la papa en Cartago, Costa Rica')
st.markdown('#### AgroIA es un sistema inteligente en Python que apoya a agricultores en Costa Rica. '
                'Usa IA (ARR, CNN, RNN) para detectar enfermedades en hojas, predecir rendimientos y '
                'recomendar acciones (riego, fertilización, poda), promoviendo una agricultura más productiva y sostenible.')
visualizador.cargar_formulario()
