#%%
# 1. Importación de librerías
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import tensorflow as tf
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization, Bidirectional
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l1_l2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator
import warnings
warnings.filterwarnings('ignore')
import os
from src.utils.metrics import obtener_ruta_app
#%%
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

print("TensorFlow version:", tf.__version__)
#%%
def cargar_datos(archivo_csv):
    """
    Carga los datos manteniendo la estructura multivariada
    """
    df = pd.read_csv(archivo_csv, parse_dates=['fecha'])
    df = df.sort_values(['fecha', 'canton_id'])  # Ordenar por fecha y cantón
    print(f"Datos cargados: {df.shape[0]} filas, {df.shape[1]} columnas")
    print(f"Rango de fechas: {df['fecha'].min()} a {df['fecha'].max()}")
    print(f"Cantones únicos: {df['canton_id'].nunique()}")
    return df
#%%
base_path = obtener_ruta_app("TemplateAgroIA")
file_path = os.path.join(base_path, "data", "processed", "rnn", "rnn_produccion.csv")
df = cargar_datos(file_path)
#%%
def preparar_datos_multivariado(df):
    """
    Prepara los datos para modelo multivariado con one-hot encoding
    """
    # One-hot encoding para cantón
    df_encoded = pd.get_dummies(df, columns=['canton_id'], prefix='canton')

    # Separar features y target
    canton_cols = [col for col in df_encoded.columns if col.startswith('canton_')]

    # Definir features (excluir fecha y produccion)
    feature_cols = [col for col in df_encoded.columns
                   if col not in ['fecha', 'produccion']]

    # Crear dataset con todas las features
    X = df_encoded[feature_cols].values
    y = df_encoded['produccion'].values.reshape(-1, 1)  # Solo producción

    print(f"Shape de X (features): {X.shape}")
    print(f"Shape de y (target): {y.shape}")
    print(f"Features incluidas: {feature_cols}")

    return X, y, feature_cols, df_encoded
#%%
X, y, feature_cols, df_encoded = preparar_datos_multivariado(df)
#%%
def dividir_train_test_multivariado_v2(X, y, df_encoded, meses_test=12):
    """
    Divide datos multivariados manteniendo estructura temporal
    """
    fechas_unicas = sorted(df_encoded['fecha'].unique())
    split_date = fechas_unicas[-meses_test]

    # Obtener índices de división
    train_idx = df_encoded['fecha'] < split_date
    test_idx = df_encoded['fecha'] >= split_date

    X_train = X[train_idx]
    X_test = X[test_idx]
    y_train = y[train_idx]
    y_test = y[test_idx]

    print(f"Datos de entrenamiento: {X_train.shape}, {y_train.shape}")
    print(f"Datos de prueba: {X_test.shape}, {y_test.shape}")

    return X_train, X_test, y_train, y_test, split_date
#%%
X_train, X_test, y_train, y_test, split_date = dividir_train_test_multivariado_v2(
        X, y, df_encoded, meses_test=12)
#%%
def escalar_datos_multivariado_v2(X_train, X_test, y_train, y_test):
    """
    Escala features y target por separado
    """
    # Escalar features
    scaler_X = MinMaxScaler()
    X_train_scaled = scaler_X.fit_transform(X_train)
    X_test_scaled = scaler_X.transform(X_test)

    # Escalar target
    scaler_y = MinMaxScaler()
    y_train_scaled = scaler_y.fit_transform(y_train)
    y_test_scaled = scaler_y.transform(y_test)

    return X_train_scaled, X_test_scaled, y_train_scaled, y_test_scaled, scaler_X, scaler_y
#%%
X_train_scaled, X_test_scaled, y_train_scaled, y_test_scaled, scaler_X, scaler_y = escalar_datos_multivariado_v2(
        X_train, X_test, y_train, y_test)
#%%
def crear_secuencias_multivariadas(X_scaled, y_scaled, length=12):
    """
    Crea secuencias temporales para modelo multivariado
    """
    X_sequences = []
    y_sequences = []

    for i in range(length, len(X_scaled)):
        X_sequences.append(X_scaled[i-length:i])
        y_sequences.append(y_scaled[i])

    return np.array(X_sequences), np.array(y_sequences)
#%%
length = 12
X_train_seq, y_train_seq = crear_secuencias_multivariadas(X_train_scaled, y_train_scaled, length)
X_test_seq, y_test_seq = crear_secuencias_multivariadas(X_test_scaled, y_test_scaled, length)

print(f"Shape de secuencias de entrenamiento: {X_train_seq.shape}, {y_train_seq.shape}")
print(f"Shape de secuencias de prueba: {X_test_seq.shape}, {y_test_seq.shape}")
#%%
def crear_modelo_lstm_multivariado(input_shape, units=128, dropout_rate=0.2):
    """
    Crea modelo LSTM multivariado que predice solo producción
    """
    model = Sequential([
        # Primera capa LSTM bidireccional
        Bidirectional(LSTM(units, return_sequences=True, input_shape=input_shape)),
        BatchNormalization(),
        Dropout(dropout_rate),

        # Segunda capa LSTM bidireccional
        Bidirectional(LSTM(units//2, return_sequences=True)),
        BatchNormalization(),
        Dropout(dropout_rate),

        # Tercera capa LSTM
        LSTM(units//2, return_sequences=False),
        BatchNormalization(),
        Dropout(dropout_rate),

        # Capas densas
        Dense(50, activation='relu', kernel_regularizer=l1_l2(l1=0.01, l2=0.01)),
        Dropout(dropout_rate),
        Dense(25, activation='relu'),
        Dense(1)  # Solo una salida: producción
    ])

    optimizer = Adam(learning_rate=0.0005)
    model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])

    return model
#%%
input_shape = (length, X_train_seq.shape[2])  # (timesteps, features)
model = crear_modelo_lstm_multivariado(input_shape)

print(f"Input shape: {input_shape}")
model.summary()
#%%
def entrenar_modelo_multivariado(model, X_train_seq, y_train_seq, X_test_seq, y_test_seq,
                                epochs=150, batch_size=32, patience=20):
    """
    Entrena el modelo multivariado
    """
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=patience,
        restore_best_weights=True
    )

    history = model.fit(
        X_train_seq, y_train_seq,
        validation_data=(X_test_seq, y_test_seq),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[early_stopping],
        verbose=1
    )

    return history
#%%
history = entrenar_modelo_multivariado(
        model, X_train_seq, y_train_seq, X_test_seq, y_test_seq
    )
#%%
def evaluar_modelo_multivariado_v2(model, X_test_seq, y_test_seq, scaler_y):
    """
    Evalúa el modelo multivariado
    """
    # Predicciones
    predictions_scaled = model.predict(X_test_seq)

    # Desescalar
    predictions = scaler_y.inverse_transform(predictions_scaled)
    real_values = scaler_y.inverse_transform(y_test_seq)

    # Métricas
    mae = mean_absolute_error(real_values, predictions)
    mse = mean_squared_error(real_values, predictions)
    r2 = r2_score(real_values, predictions)

    print(f"MAE: {mae:.4f}")
    print(f"MSE: {mse:.4f}")
    print(f"R²: {r2:.4f}")

    return predictions, real_values
#%%
predictions, real_values = evaluar_modelo_multivariado_v2(
        model, X_test_seq, y_test_seq, scaler_y)
#%%
def forecast_futuro_multivariado(model, X_scaled, scaler_X, scaler_y, length=12, pasos_futuros=6):
    """
    Realiza predicciones futuras con modelo multivariado
    NOTA: Requiere valores futuros de las features (área, clima, cantón)
    """
    # Obtener la última secuencia
    last_sequence = X_scaled[-length:]

    forecast = []
    current_sequence = last_sequence.copy()

    for _ in range(pasos_futuros):
        # Predecir siguiente valor
        next_pred = model.predict(current_sequence.reshape(1, length, -1), verbose=0)
        forecast.append(next_pred[0, 0])

        # Para forecast multivariado, necesitarías actualizar current_sequence
        # con nuevos valores de features. Aquí simplificamos manteniendo
        # las features constantes y solo actualizando la secuencia temporal

        # Actualizar secuencia (esto es una simplificación)
        current_sequence = np.roll(current_sequence, -1, axis=0)
        # current_sequence[-1] = ... # Aquí necesitarías los valores futuros de features

    # Desescalar predicciones
    forecast = np.array(forecast).reshape(-1, 1)
    forecast_unscaled = scaler_y.inverse_transform(forecast)

    return forecast_unscaled.flatten()
#%%
model_path = os.path.join(base_path, "models", "modelo_LSTM_multivariado.keras")
model.save(model_path)
print("Modelo guardado con éxito.")
#%%
def guardar_scalers_y_features(scaler_X, scaler_y, feature_cols, base_path):
    """
    Guarda los scalers y columnas de features para uso posterior
    """
    # Crear directorio models si no existe
    models_dir = os.path.join(base_path, "models")
    os.makedirs(models_dir, exist_ok=True)

    # Guardar scaler de features (X)
    scaler_X_path = os.path.join(models_dir, "scaler_X_multivariado.pkl")
    with open(scaler_X_path, 'wb') as f:
        pickle.dump(scaler_X, f)

    # Guardar scaler de target (y)
    scaler_y_path = os.path.join(models_dir, "scaler_y_multivariado.pkl")
    with open(scaler_y_path, 'wb') as f:
        pickle.dump(scaler_y, f)

    # Guardar columnas de features
    features_path = os.path.join(models_dir, "feature_columns_multivariado.pkl")
    with open(features_path, 'wb') as f:
        pickle.dump(feature_cols, f)

    print("✅ Scalers y features guardados correctamente.")
    print(f"   - Scaler X: {scaler_X_path}")
    print(f"   - Scaler y: {scaler_y_path}")
    print(f"   - Features: {features_path}")

#%%
guardar_scalers_y_features(scaler_X, scaler_y, feature_cols, base_path)
#%%
def visualizar_entrenamiento(history):
    """
    Visualiza las curvas de entrenamiento y validación
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    # Gráfico de pérdida
    ax1.plot(history.history['loss'], label='Entrenamiento', linewidth=2)
    ax1.plot(history.history['val_loss'], label='Validación', linewidth=2)
    ax1.set_title('Pérdida del Modelo', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Época')
    ax1.set_ylabel('Pérdida (MSE)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Gráfico de MAE
    ax2.plot(history.history['mae'], label='Entrenamiento', linewidth=2)
    ax2.plot(history.history['val_mae'], label='Validación', linewidth=2)
    ax2.set_title('Error Absoluto Medio', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Época')
    ax2.set_ylabel('MAE')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    # Imprimir estadísticas finales
    print("\n=== ESTADÍSTICAS FINALES ===")
    print(f"Pérdida final (entrenamiento): {history.history['loss'][-1]:.4f}")
    print(f"Pérdida final (validación): {history.history['val_loss'][-1]:.4f}")
    print(f"MAE final (entrenamiento): {history.history['mae'][-1]:.4f}")
    print(f"MAE final (validación): {history.history['val_mae'][-1]:.4f}")
    print(f"Épocas entrenadas: {len(history.history['loss'])}")

#%%
visualizar_entrenamiento(history)
#%% md
# ### PRUEBA DEL MODELO
# ### Cargar modelo y scalers
#%%
model_path = os.path.join(base_path, "models", "modelo_LSTM_multivariado.keras")
scaler_y_path = os.path.join(base_path, "models", "scaler_y_multivariado.pkl")
scaler_X_path = os.path.join(base_path, "models", "scaler_X_multivariado.pkl")
features_path = os.path.join(base_path, "models", "feature_columns_multivariado.pkl")

modelo_cargado = tf.keras.models.load_model(model_path)
with open(scaler_y_path, 'rb') as f:
    scaler_y_cargado = pickle.load(f)
with open(scaler_X_path, 'rb') as f:
    scaler_X_cargado = pickle.load(f)
with open(features_path, 'rb') as f:
    columnas_features = pickle.load(f)

print("✅ Modelo y scalers cargados")
#%% md
# ### Cargar datos de prueba
#%%
test_file_path = os.path.join(base_path, "data", "raw", "df_prueba_produccion.csv")
df_test = pd.read_csv(test_file_path, parse_dates=['fecha'])

# Tomar últimos 12 meses de un cantón
canton_id = df_test['canton_id'].iloc[0]
df_test = df_test[df_test['canton_id'] == canton_id].tail(12)

print(f"Usando {len(df_test)} registros del cantón {canton_id}")
#%% md
# ### Preparar datos
#%%
df_encoded = pd.get_dummies(df_test, columns=['canton_id'], prefix='canton')
for col in columnas_features:
    if col not in df_encoded.columns:
        df_encoded[col] = 0
df_encoded = df_encoded[columnas_features]

X_scaled = scaler_X_cargado.transform(df_encoded.values)
#%% md
# ### Hacer predicción
#%%
meses_a_predecir = 6
predicciones = []
current_seq = X_scaled.copy()

for i in range(meses_a_predecir):
    pred = modelo_cargado.predict(current_seq.reshape(1, 12, -1), verbose=0)
    predicciones.append(pred[0, 0])
    # Rotar secuencia para siguiente predicción
    current_seq = np.roll(current_seq, -1, axis=0)
#%% md
# ### Desescalar predicciones
#%%
predicciones_reales = scaler_y_cargado.inverse_transform(np.array(predicciones).reshape(-1, 1)).flatten()

print(f"Predicciones para los próximos {meses_a_predecir} meses:")
for i, pred in enumerate(predicciones_reales, 1):
    print(f"Mes {i}: {pred:.2f} unidades")
#%% md
# ### Gráfico con múltiples predicciones
#%%
fechas_futuras = pd.date_range(start=df_test['fecha'].max() + pd.DateOffset(months=1),
                              periods=meses_a_predecir, freq='ME')

plt.figure(figsize=(12, 6))
plt.plot(df_test['fecha'], df_test['produccion'], 'o-', label='Histórico', linewidth=2)
plt.plot(fechas_futuras, predicciones_reales, 's-', color='red', label='Predicciones', linewidth=2)
plt.axvline(x=df_test['fecha'].max(), color='gray', linestyle='--', alpha=0.5)
plt.title('Predicción de Producción - Múltiples Meses')
plt.xlabel('Fecha')
plt.ylabel('Producción')
plt.legend()
plt.grid(True, alpha=0.3)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

print(f"Producción promedio histórica: {df_test['produccion'].mean():.2f}")
print(f"Predicción promedio: {np.mean(predicciones_reales):.2f}")
print(f"Diferencia promedio: {np.mean(predicciones_reales) - df_test['produccion'].mean():.2f}")