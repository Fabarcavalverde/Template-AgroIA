import streamlit as st
import pandas as pd
from src.train.rnn import rnn
from src.utils.metrics import obtener_ruta_app

class Modulo2Web:
    def __init__(self):
        self.modelo = rnn(obtener_ruta_app("Template-AgroIA"))


    def render(self):
        st.set_page_config(page_title="Predicción de Producción", layout="centered")
        st.title("Predicción de Producción de Papa")
        archivo = st.file_uploader("Sube un archivo .csv o .xlsx", type=["csv", "xlsx"])

        if archivo:
            try:
                if archivo.name.endswith(".csv"):
                    df = pd.read_csv(archivo)
                else:
                    df = pd.read_excel(archivo)

                st.write("Vista previa del archivo:")
                st.dataframe(df)

                pred = self.modelo.predecir_df(
                    df
                )

                st.success(f"Predicción del próximo mes: *{pred:.2f} toneladas*")

            except Exception as e:
                st.error(f" Error: {e}")