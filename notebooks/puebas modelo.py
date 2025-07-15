import numpy as np
import pandas as pd
import pickle
from tensorflow.keras.models import load_model
import os
from src.utils.metrics import obtener_ruta_app
from keras.losses import MeanSquaredError

base_path = obtener_ruta_app("TemplateAgroIA")
model_path = os.path.join(base_path, "models", "modelo_LSTM_multivariado.keras")
scaler_y_path = os.path.join(base_path, "models", "scaler_y_multivariado.pkl")
scaler_X_path = os.path.join(base_path, "models", "scaler_X_multivariado.pkl")
features_path = os.path.join(base_path, "models", "feature_columns_multivariado.pkl")

# Cargar modelo y scalers
model = load_model(model_path, custom_objects={"mse": MeanSquaredError})
print("✅ Modelo cargado.")

with open(scaler_y_path, 'rb') as f:
    scaler_y = pickle.load(f)
print("✅ Scaler y cargado.")

with open(scaler_X_path, 'rb') as f:
    scaler_X = pickle.load(f)
print("✅ Scaler X cargado.")

with open(features_path, 'rb') as f:
    columnas_features = pickle.load(f)
print("✅ Columnas de features cargadas.")

# Crear dataframe de prueba (últimos 12 meses conocidos)
df_prueba = pd.DataFrame({
    'area': np.random.randint(90, 120, size=12),
    'PRECTOTCORR_SUM': np.random.randint(200, 300, size=12),
    'RH2M': np.random.uniform(70, 90, size=12),
    'T2M': np.random.uniform(20, 25, size=12),
    'T2MDEW': np.random.uniform(17, 19, size=12),
    'T2MWET': np.random.uniform(18, 20, size=12),
    'T2M_MAX': np.random.uniform(28, 31, size=12),
    'T2M_MIN': np.random.uniform(14, 17, size=12),
    'WS2M': np.random.uniform(0.8, 1.5, size=12),
    'canton_id': [205] * 12,
    'mes': np.arange(1, 13),
    'produccion': np.random.randint(1100, 1300, size=12)
})
df_prueba['fecha'] = pd.date_range('2023-01-01', periods=12, freq='ME')

# One-hot encoding y alinear columnas
df_encoded = pd.get_dummies(df_prueba, columns=['canton_id'], prefix='canton')
for col in columnas_features:
    if col not in df_encoded.columns:
        df_encoded[col] = 0
df_encoded = df_encoded[columnas_features]

# Escalar features y target
X_scaled = scaler_X.transform(df_encoded.values)
y_scaled = scaler_y.transform(df_prueba['produccion'].values.reshape(-1, 1))

length = 12  # longitud de la secuencia
pasos_futuros = 6  # meses a predecir


def forecast_multistep(model, X_scaled, scaler_y, length=12, pasos_futuros=6):
    """
    Forecast múltiple con secuencia rolling.
    Mantiene features constantes para los pasos futuros (simplificación).
    """
    current_seq = X_scaled[-length:].copy()  # última secuencia

    preds_scaled = []

    for _ in range(pasos_futuros):
        # Predecir siguiente producción escalada
        pred_scaled = model.predict(current_seq.reshape(1, length, -1), verbose=0)[0, 0]
        preds_scaled.append(pred_scaled)

        # Construir siguiente secuencia
        # Desplazar ventana 1 paso hacia adelante
        current_seq = np.roll(current_seq, -1, axis=0)

        # Mantener features constantes excepto producción que no está en features
        # NOTA: producción no es parte de X_scaled, entonces no la actualizamos acá

        # Si producción fuera parte de features, actualizaríamos la última fila con la predicción
        # Pero aquí, las features permanecen constantes

    preds_scaled = np.array(preds_scaled).reshape(-1, 1)
    preds_real = scaler_y.inverse_transform(preds_scaled)

    return preds_real.flatten()


predicciones_futuras = forecast_multistep(model, X_scaled, scaler_y, length, pasos_futuros)

print(f"📈 Predicciones de producción para los próximos {pasos_futuros} meses:")
for i, pred in enumerate(predicciones_futuras, 1):
    print(f"Mes {i}: {pred:.2f} unidades")

# Guardar predicciones en CSV
df_output = pd.DataFrame({
    "mes_futuro": np.arange(1, pasos_futuros + 1),
    "prediccion_produccion": predicciones_futuras
})
result_path = os.path.join(base_path, "results", "prediccion_test_LSTM_multistep.csv")
os.makedirs(os.path.dirname(result_path), exist_ok=True)
df_output.to_csv(result_path, index=False)
print(f"✅ Predicciones guardadas en: {result_path}")
