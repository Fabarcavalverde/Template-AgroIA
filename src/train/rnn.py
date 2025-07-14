"""
Clase: rnn_produccion

Objetivo: Clase para cargar modelo RNN entrenado y realizar predicciones a partir de archivo CSV o Excel.

Cambios:
    1. Creación de clase basada en ejemplo CNN - Fiorella, 14-07-2025
"""
# src/train/rnn.py
import os
import numpy as np
import pandas as pd
import pickle
from tensorflow.keras.models import load_model
from keras.losses import mean_squared_error

class RNNProduccion:
    def __init__(self, ruta_base):
        self.ventana_tiempo = 3
        self.columnas_features = [
            'area', 'PRECTOTCORR_SUM', 'RH2M', 'T2M', 'T2MDEW',
            'T2MWET', 'T2M_MAX', 'T2M_MIN', 'WS2M', 'mes'
        ]
        self.ruta_modelo = os.path.join(ruta_base, "models", "modelo_RNN_Papas.h5")
        self.ruta_scaler = os.path.join(ruta_base, "models", "scaler_RNN_Papas.pkl")

        self.model = load_model(self.ruta_modelo, custom_objects={'mse': mean_squared_error})
        with open(self.ruta_scaler, 'rb') as f:
            self.scaler = pickle.load(f)

    def predecir_df(self, df: pd.DataFrame) -> float:
        df = df.copy()
        df['fecha'] = pd.to_datetime(df['fecha'])
        df = df.sort_values("fecha")

        if len(df) < self.ventana_tiempo:
            raise ValueError(f"Se requieren al menos {self.ventana_tiempo} filas")

        data = df[self.columnas_features + ['produccion']].values
        data_scaled = self.scaler.transform(data)
        X = data_scaled[-self.ventana_tiempo:, :-1]
        X = np.expand_dims(X, axis=0)

        pred_scaled = self.model.predict(X, verbose=0)[0][0]
        temp = np.zeros((1, self.scaler.n_features_in_))
        temp[0, -1] = pred_scaled
        pred_real = self.scaler.inverse_transform(temp)[0, -1]
        return pred_real
