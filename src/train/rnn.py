"""
Clase: rnn_produccion

Objetivo: Clase para cargar modelo RNN entrenado y realizar predicciones a partir de archivo CSV o Excel.

Cambios:
    1. Creación de clase basada en ejemplo CNN - Fiorella, 14-07-2025
"""
import os
import numpy as np
import pandas as pd
import pickle
from tensorflow.keras.models import load_model
from keras.losses import mean_squared_error
from src.utils.metrics import obtener_ruta_app


class rnn:
    def __init__(self, ruta_raiz: str):
        self.ventana_tiempo = 3  # <- Esta línea es la clave
        self.ruta_modelo = os.path.join(ruta_raiz, "models", "modelo_RNN_Papas.h5")
        self.ruta_scaler = os.path.join(ruta_raiz, "models", "scaler_RNN_Papas.pkl")
        self.columnas_features = ['area', 'PRECTOTCORR_SUM', 'RH2M', 'T2M', 'T2MDEW',
                                  'T2MWET', 'T2M_MAX', 'T2M_MIN', 'WS2M', 'mes']

        self.model = load_model(self.ruta_modelo, custom_objects={'mse': mean_squared_error})
        with open(self.ruta_scaler, 'rb') as f:
            self.scaler = pickle.load(f)

        print("✅ Modelo y scaler cargados correctamente.")


    def _leer_archivo(self, ruta_archivo: str) -> pd.DataFrame:
        if ruta_archivo.endswith('.csv'):
            df = pd.read_csv(ruta_archivo)
        elif ruta_archivo.endswith('.xlsx'):
            df = pd.read_excel(ruta_archivo)
        else:
            raise ValueError("El archivo debe ser .csv o .xlsx")
        if 'fecha' not in df.columns:
            raise ValueError("El archivo debe contener una columna 'fecha'")
        df['fecha'] = pd.to_datetime(df['fecha'])
        return df

    def _preparar_input(self, df: pd.DataFrame) -> np.ndarray:
        df = df.sort_values("fecha")
        if len(df) < self.ventana_tiempo:
            raise ValueError(f"Se requieren al menos {self.ventana_tiempo} registros con meses consecutivos.")

        data = df[self.columnas_features + ['produccion']].values
        data_scaled = self.scaler.transform(data)
        X = data_scaled[-self.ventana_tiempo:, :-1]  # Últimos 3 meses de features
        return np.expand_dims(X, axis=0)

    def predecir_archivo(self, ruta_archivo: str) -> float:
        df = self._leer_archivo(ruta_archivo)
        X = self._preparar_input(df)

        pred_scaled = self.model.predict(X, verbose=0)[0][0]

        # Desnormalizar solo la producción
        temp = np.zeros((1, self.scaler.n_features_in_))
        temp[0, -1] = pred_scaled
        pred_real = self.scaler.inverse_transform(temp)[0, -1]

        print(f"📈 Predicción del próximo mes: {pred_real:.2f} toneladas")
        return pred_real