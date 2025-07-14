#%%
# 1. Importación de librerías
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
    Carga los datos desde un archivo CSV con fecha como índice
    """
    df = pd.read_csv(archivo_csv, parse_dates=['fecha'], index_col='fecha')
    df = df.sort_index()  # Asegurar orden cronológico
    print(f"Datos cargados: {df.shape[0]} filas, {df.shape[1]} columnas")
    print(f"Rango de fechas: {df.index.min()} a {df.index.max()}")
    return df
#%%
base_path = obtener_ruta_app("TemplateAgroIA")
file_path = os.path.join(base_path, "data", "processed", "rnn", "rnn_produccion.csv")
df = cargar_datos(file_path)
#%%
def visualizar_serie(df, columna='produccion'):
    """
    Visualiza la serie temporal de producción
    """
    plt.figure(figsize=(15, 6))
    plt.plot(df.index, df[columna], linewidth=2)
    plt.title(f'Serie Temporal de {columna.capitalize()}', fontsize=16)
    plt.xlabel('Fecha', fontsize=12)
    plt.ylabel(f'{columna.capitalize()}', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    # Estadísticas descriptivas
    print(f"\nEstadísticas de {columna}:")
    print(df[columna].describe())
#%%
visualizar_serie(df)
#%%
def dividir_train_test(df, columna='produccion', meses_test=48):
    """
    Divide los datos en entrenamiento y prueba
    """
    serie = df[columna].values.reshape(-1, 1)

    # Punto de corte
    split_point = len(serie) - meses_test

    train_data = serie[:split_point]
    test_data = serie[split_point:]

    print(f"Datos de entrenamiento: {len(train_data)} meses")
    print(f"Datos de prueba: {len(test_data)} meses")

    return train_data, test_data, split_point
#%%
train_data, test_data, split_point = dividir_train_test(df, meses_test=12)

#%%
def escalar_datos(train_data, test_data):
    """
    Escala los datos usando MinMaxScaler
    """
    scaler = MinMaxScaler()
    train_scaled = scaler.fit_transform(train_data)
    test_scaled = scaler.transform(test_data)

    return train_scaled, test_scaled, scaler

#%%
train_scaled, test_scaled, scaler = escalar_datos(train_data, test_data)
#%%
def crear_generadores(train_scaled, test_scaled, length=6, batch_size=32):
    """
    Crea generadores de secuencias temporales
    """
    train_generator = TimeseriesGenerator(
        train_scaled, train_scaled,
        length=length,
        batch_size=batch_size
    )

    test_generator = TimeseriesGenerator(
        test_scaled, test_scaled,
        length=length,
        batch_size=batch_size
    )

    return train_generator, test_generator
#%%
length = 11 # Usar 12 meses anteriores para predecir
train_generator, test_generator = crear_generadores(train_scaled, test_scaled, length=length)
#%%
def crear_modelo_lstm(input_shape, units=128, dropout_rate=0.2):
    model = Sequential([
        # Bidirectional LSTM para capturar patrones en ambas direcciones
        Bidirectional(LSTM(units, return_sequences=True, input_shape=input_shape)),
        BatchNormalization(),
        Dropout(dropout_rate),

        Bidirectional(LSTM(units//2, return_sequences=True)),
        BatchNormalization(),
        Dropout(dropout_rate),

        LSTM(units//2, return_sequences=False),
        BatchNormalization(),
        Dropout(dropout_rate),

        Dense(50, activation='relu', kernel_regularizer=l1_l2(l1=0.01, l2=0.01)),
        Dropout(dropout_rate),
        Dense(25, activation='relu'),
        Dense(1)
    ])

    # Optimizador con learning rate personalizado
    optimizer = Adam(learning_rate=0.0005)
    model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])

    return model
#%%
model = crear_modelo_lstm(input_shape=(length, 1),)
print("\nArquitectura del modelo:")
model.summary()
#%%
def entrenar_modelo(model, train_generator, test_generator, epochs, patience=10):
    """
    Entrena el modelo con EarlyStopping
    """
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=patience,
        restore_best_weights=True
    )

    history = model.fit(
        train_generator,
        epochs=epochs,
        validation_data=test_generator,
        callbacks=[early_stopping],
        verbose=1
    )

    return history
#%%
print("\nIniciando entrenamiento...")
history = entrenar_modelo(model, train_generator, test_generator, epochs=150, patience=20)
#%%
def evaluar_modelo(model, test_generator, scaler, df, split_point):
    """
    Evalúa el modelo y genera predicciones
    """
    # Predicciones
    predictions_scaled = model.predict(test_generator)
    predictions = scaler.inverse_transform(predictions_scaled)

    # Valores reales (ajustar por el desfase del generador)
    length = test_generator.length
    real_values = df['produccion'].iloc[split_point + length:].values

    # Métricas
    mae = mean_absolute_error(real_values, predictions.flatten())
    mse = mean_squared_error(real_values, predictions.flatten())
    rmse = np.sqrt(mse)
    r2 = r2_score(real_values, predictions.flatten())

    print(f"\nMétricas de evaluación:")
    print(f"MAE: {mae:.4f}")
    print(f"MSE: {mse:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"R²: {r2:.4f}")

    return predictions, real_values
#%%
predictions, real_values = evaluar_modelo(model, test_generator, scaler, df, split_point)
#%%
print(df.tail(20)[['produccion']])
#%%
#Guardar el modelo
base_path = obtener_ruta_app("TemplateAgroIA")
model_path = os.path.join(base_path, "models", "modelo_RNN_Papas.h5")
print("Model path:", model_path)
# Guarda el modelo
model.save(model_path)
print("Modelo guardado con éxito.")
#%%
def forecast_futuro(model, df, scaler, length=12, pasos_futuros=6):
    """
    Realiza predicciones futuras
    """
    # Obtener los últimos valores para iniciar el forecast
    last_sequence = df['produccion'].tail(length).values.reshape(-1, 1)
    last_sequence_scaled = scaler.transform(last_sequence)

    forecast = []
    current_sequence = last_sequence_scaled.copy()

    for _ in range(pasos_futuros):
        # Predecir siguiente valor
        next_pred = model.predict(current_sequence.reshape(1, length, 1), verbose=0)
        forecast.append(next_pred[0, 0])

        # Actualizar secuencia
        current_sequence = np.roll(current_sequence, -1)
        current_sequence[-1] = next_pred

    # Desescalar predicciones
    forecast = np.array(forecast).reshape(-1, 1)
    forecast_unscaled = scaler.inverse_transform(forecast)

    return forecast_unscaled.flatten()
#%%
forecast = forecast_futuro(model, df, scaler, length=length, pasos_futuros=6)
print(f"\nForecast para los próximos 6 meses:")
for i, pred in enumerate(forecast, 1):
    print(f"Mes {i}: {pred:.2f}")
#%%

def graficar_predicciones(predictions, real_values, forecast, df, split_point, length=36, pasos_futuros=6):
    """
    Grafica solo las predicciones, valores reales y forecast
    """
    plt.figure(figsize=(16, 8))

    # Fechas para predicciones en test
    test_dates = df.index[split_point + length:]

    # Solo predicciones vs reales
    plt.plot(test_dates, real_values, label='Valores Reales', color='green', linewidth=3, marker='o')
    plt.plot(test_dates, predictions.flatten(), label='Predicciones', color='red', linewidth=3, marker='s')

    # Forecast futuro
    last_date = df.index[-1]
    future_dates = pd.date_range(start=last_date + pd.DateOffset(months=1), periods=pasos_futuros, freq='MS')
    plt.plot(future_dates, forecast, label='Forecast Futuro', color='orange', linewidth=3, linestyle='--', marker='^')

    plt.title('Predicciones vs Valores Reales - Modelo LSTM', fontsize=16)
    plt.xlabel('Fecha', fontsize=12)
    plt.ylabel('Producción', fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
#%%
graficar_predicciones(predictions, real_values, forecast, df, split_point, length, pasos_futuros=6)
#%%
# Graficar curvas de entrenamiento
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Pérdida del Modelo')
plt.xlabel('Época')
plt.ylabel('Loss')
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
plt.plot(history.history['mae'], label='Train MAE')
plt.plot(history.history['val_mae'], label='Validation MAE')
plt.title('Error Absoluto Medio')
plt.xlabel('Época')
plt.ylabel('MAE')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

#%%


#%%
plt.figure(figsize=(15,5))
plt.plot(df.index, df['produccion'], label='Total')
plt.axvline(df.index[split_point], color='red', linestyle='--', label='Corte Train/Test')
plt.legend()
plt.title("Separación Entrenamiento vs Prueba")
plt.show()
