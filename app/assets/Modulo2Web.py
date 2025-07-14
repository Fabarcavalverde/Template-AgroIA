import streamlit as st
import pandas as pd


class Modulo2Web:
    def __init__(self):
        pass

    def cargar_formulario(self):
        st.title("Predicción del rendimiento de cultivo de papa")
        CANTON_MAP = {
            "Turrialba": 1.0,
            "Oreamuno": 0.75,
            "El Guarco": 0.5,
            "Cartago": 0.25,
            "Alvarado": 0.0,
        }
        st.markdown("#### Datos del cultivo")

        c1, c2, c3 = st.columns(3)
        canton_nombre = c1.selectbox("Cantón", list(CANTON_MAP.keys()))
        canton_id = CANTON_MAP[canton_nombre]  # ← ya es float/numérico
        # puedes cambiar a nombres después
        area = c2.number_input("Área cultivada (ha)", min_value=0.0, max_value=100.0, value=0.5)
        fecha = c3.date_input("Fecha del registro")

        st.markdown("#### Variables atmosféricas (MERRA-2)")

        c4, c5, c6 = st.columns(3)
        prec = c4.number_input("Precipitación corregida (mm/día)", min_value=0.0, value=5.0, key="prec")
        humedad = c5.number_input("Humedad relativa 2 m (%)", min_value=0.0, max_value=100.0, value=75.0, key="humedad")
        temp = c6.number_input("Temperatura 2 m (°C)", min_value=-10.0, max_value=50.0, value=20.0, key="temp")

        c7, c8, c9 = st.columns(3)
        dew = c7.number_input("Punto de rocío 2 m (°C)", min_value=-10.0, max_value=40.0, value=15.0, key="dew")
        wet = c8.number_input("Bulbo húmedo 2 m (°C)", min_value=-10.0, max_value=40.0, value=17.0, key="wet")
        ws2m = c9.number_input("Viento 2 m (m/s)", min_value=0.0, max_value=20.0, value=2.5, key="ws2m")

        c10, c11 = st.columns(2)
        tmax = c10.number_input("Temp. máxima 2 m (°C)", min_value=-10.0, max_value=60.0, value=28.0, key="tmax")
        tmin = c11.number_input("Temp. mínima 2 m (°C)", min_value=-10.0, max_value=40.0, value=12.0, key="tmin")


        # Mostrar valores por ahora
        if st.button("Ver entrada"):
            st.write("Datos ingresados:")
            st.json({
                "canton_id": canton_id,
                "area": area,
                "fecha": str(fecha),
                "PRECTOTCORR_SUM": prec,
                "RH2M": humedad,
                "T2M": temp,
                "T2MDEW": dew,
                "T2MWET": wet,
                "T2M_MAX": tmax,
                "T2M_MIN": tmin,
                "WS2M": ws2m
            })
