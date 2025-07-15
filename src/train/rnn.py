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
from keras.losses import MeanSquaredError


class rnn:
    """
    Clase para cargar modelo LSTM multivariado y realizar predicciones multistep
    de producción agrícola a partir de DataFrames.

    Cambios:
        Basada en estructura ejemplo CNN y código de predicción multistep con rolling window.
        Fiorella, 15-07-2025
    """

    def __init__(self, ruta_base):
        self.length = 12  # longitud secuencia para entrada
        self.pasos_futuros = 6  # meses a predecir

        self.ruta_modelo = os.path.join(ruta_base, "models", "modelo_LSTM_multivariado.keras")
        self.ruta_scaler_y = os.path.join(ruta_base, "models", "scaler_y_multivariado.pkl")
        self.ruta_scaler_X = os.path.join(ruta_base, "models", "scaler_X_multivariado.pkl")
        self.ruta_columnas = os.path.join(ruta_base, "models", "feature_columns_multivariado.pkl")

        # Cargar modelo
        self.model = load_model(self.ruta_modelo, custom_objects={"mse": MeanSquaredError})
        print("✅ Modelo cargado.")

        # Cargar scalers
        with open(self.ruta_scaler_y, 'rb') as f:
            self.scaler_y = pickle.load(f)
        print("✅ Scaler y cargado.")

        with open(self.ruta_scaler_X, 'rb') as f:
            self.scaler_X = pickle.load(f)
        print("✅ Scaler X cargado.")

        # Cargar columnas features
        with open(self.ruta_columnas, 'rb') as f:
            self.columnas_features = pickle.load(f)
        print("✅ Columnas de features cargadas.")

    def predecir_multistep(self, df: pd.DataFrame) -> np.ndarray:
        """
        Recibe un DataFrame con los datos históricos y realiza predicción para los próximos
        pasos_futuros meses usando rolling window.
        """

        df = df.copy()
        # Asegurar que fecha sea datetime
        if 'fecha' in df.columns:
            df['fecha'] = pd.to_datetime(df['fecha'])
        else:
            raise ValueError("El DataFrame debe contener columna 'fecha'")

        # One-hot encode 'canton_id' para alinear con columnas features
        df_encoded = pd.get_dummies(df, columns=['canton_id'], prefix='canton')

        # Asegurar que todas las columnas de features existan
        for col in self.columnas_features:
            if col not in df_encoded.columns:
                df_encoded[col] = 0

        # Ordenar columnas para que coincidan con el scaler
        df_encoded = df_encoded[self.columnas_features]

        # Escalar features
        X_scaled = self.scaler_X.transform(df_encoded.values)

        # Validar que la secuencia sea suficiente para la ventana
        if X_scaled.shape[0] < self.length:
            raise ValueError(f"Se requieren al menos {self.length} filas para la predicción.")

        # Forecast multistep con rolling window
        current_seq = X_scaled[-self.length:].copy()
        preds_scaled = []

        for _ in range(self.pasos_futuros):
            # Predecir producción escalada
            pred_scaled = self.model.predict(current_seq.reshape(1, self.length, -1), verbose=0)[0, 0]
            preds_scaled.append(pred_scaled)

            # Mover ventana 1 paso adelante (rolling)
            current_seq = np.roll(current_seq, -1, axis=0)
            # Aquí mantenemos features constantes porque producción no está en features

        preds_scaled = np.array(preds_scaled).reshape(-1, 1)
        preds_real = self.scaler_y.inverse_transform(preds_scaled).flatten()

        return preds_real


