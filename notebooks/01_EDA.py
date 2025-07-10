#%%
import numpy as np
from src.PipelineProcesamiento import PipelineProcesamiento
from src.utils.metrics import obtener_ruta_app
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
#%%
ejecutar = PipelineProcesamiento(os.path.join(obtener_ruta_app("Template AgroIA"), "data/raw/rnn/ESTIM_papa_2005-2025.xls"),os.path.join(obtener_ruta_app("Template AgroIA"), "data/raw/rnn/DatosAtmosfericos"))

df= ejecutar.ejecutar_pipeline_completo()
df.head()
#%%
df.info()
#%%
meses_es_en = {
    'enero': 'January', 'febrero': 'February', 'marzo': 'March',
    'abril': 'April', 'mayo': 'May', 'junio': 'June',
    'julio': 'July', 'agosto': 'August', 'septiembre': 'September',
    'octubre': 'October', 'noviembre': 'November', 'diciembre': 'December'
}

# Crear columna de fecha
df['mes_en'] = df['mes'].map(meses_es_en)
df['fecha'] = pd.to_datetime(df['mes_en'] + ' ' + df['anio'].astype(str), format='%B %Y')

# Eliminar columnas unificadas
df.drop(columns=['mes', 'anio', 'mes_en'], inplace=True)


#%% md
# # Limpieza final
#%%
df.head
#%%
df.drop(['IMERG_PRECTOT'], axis=1,inplace=True) #Eliminar variable basura
#%%
print(df.corr(numeric_only=True)['produccion'].sort_values())
#%%
df.replace(-999, np.nan, inplace=True)
#%%
df = df[~((df['fecha'].dt.year == 2025) & (df['fecha'].dt.month > 6))]

#%%
df.hist(bins=15, figsize=(20, 16), color='#1F4E79', edgecolor='black')
#%%
df.describe()
#%%
df[['area', 'produccion']].plot.box()
plt.show()
#%%
plt.figure(figsize=(10, 3))
sns.kdeplot(df['produccion'], fill=True, color='#1F4E79', alpha=0.6)
plt.title('KDE - Minutes Elapsed', fontsize=16)
plt.xlabel('producciond', fontsize=12)
plt.ylabel('Density', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()
#%%
df.sort_values(by=['produccion'], ascending=False)
#%%
df.fillna(method='ffill', inplace=True)
#%%
print(df)
#%%
print(df.isnull().sum())
#%%

# Crear rango de fechas mensuales de enero 2005 a mayo 2025
fechas = pd.date_range(start='2005-01-01', end='2025-05-01', freq='MS')  # MS = Month Start

# Obtener lista única de cantones
cantones = df['canton'].unique()

# Crear todas las combinaciones posibles
combinaciones = pd.MultiIndex.from_product([cantones, fechas], names=['canton', 'fecha']).to_frame(index=False)

#%%
# Asegurar que Fecha en df está al inicio del mes
df['fecha'] = df['fecha'].dt.to_period('M').dt.to_timestamp()

# Crear DataFrame con los registros existentes
df_existente = df[['canton', 'fecha']].drop_duplicates()

# Hacer merge para ver cuáles combinaciones faltan
faltantes = combinaciones.merge(df_existente, on=['canton', 'fecha'], how='left', indicator=True)

# Filtrar solo las combinaciones que NO están en el df
faltantes = faltantes[faltantes['_merge'] == 'left_only'].drop(columns=['_merge'])
print(faltantes)
#%%
print(df.corr(numeric_only=True)['produccion'].sort_values())
#%%
print(df.corr(numeric_only=True)['produccion'].sort_values())
#%%
df = df.drop([
    'WS10M',
    'GWETTOP',
    'GWETPROF',
    'GWETROOT',
    'ALLSKY_SFC_SW_DWN',
    'CLRSKY_SFC_SW_DWN',
    'CLRSKY_SFC_PAR_TOT'
],axis=1)

#%%
df.plot(figsize=(12,8))
#%% raw
# Train and Test Split
#%%
#Prepara la variable categorica
df['canton_id'] = df['canton'].astype('category').cat.codes
num_cantones = df['canton_id'].nunique()

#%%
df= df.drop('canton',axis=1)
#%% raw
# Normalizar tus variables numéricas
#%%
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
features_to_scale = ['area', 'PRECTOTCORR_SUM', 'RH2M', 'T2M', 'T2MDEW',
       'T2MWET', 'T2M_MAX', 'T2M_MIN', 'WS2M', 'canton_id']  # tus variables numéricas
df[features_to_scale] = scaler.fit_transform(df[features_to_scale])

#%%
df.head()
#%%
import numpy as np

ventana = 12
X_num, X_canton, y = [], [], []

for canton_id in df['canton_id'].unique():
    df_c = df[df['canton_id'] == canton_id].sort_values('fecha')

    datos = df_c[features_to_scale].values
    canton_id_array = df_c['canton_id'].values
    target = df_c['produccion'].values

    for i in range(len(df_c) - ventana):
        X_num.append(datos[i:i+ventana])
        X_canton.append(canton_id_array[i])  # un solo valor por secuencia
        y.append(target[i+ventana])

#%%
X_num = np.array(X_num)       # shape: (samples, ventana, num_features)
X_canton = np.array(X_canton)  # shape: (samples,)
y = np.array(y)               # shape: (samples,)

#%%
import torch
from torch.utils.data import Dataset

class SecuenciasDataset(Dataset):
    def __init__(self, X_num, X_canton, y):
        self.X_num = torch.tensor(X_num, dtype=torch.float32)
        self.X_canton = torch.tensor(X_canton, dtype=torch.long)  # necesario para embedding
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X_num[idx], self.X_canton[idx], self.y[idx]

#%%
import torch.nn as nn

class RNNConEmbeddings(nn.Module):
    def __init__(self, input_size, hidden_size, num_cantones, embedding_dim):
        super(RNNConEmbeddings, self).__init__()

        self.embedding = nn.Embedding(num_cantones, embedding_dim)
        self.rnn = nn.LSTM(input_size + embedding_dim, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x_num, canton_id):
        batch_size, seq_len, _ = x_num.size()

        # Expandir embedding a toda la secuencia
        canton_embed = self.embedding(canton_id)  # (batch_size, embedding_dim)
        canton_embed = canton_embed.unsqueeze(1).expand(-1, seq_len, -1)  # (batch_size, seq_len, embedding_dim)

        # Concatenar a variables numéricas
        x = torch.cat([x_num, canton_embed], dim=2)  # (batch_size, seq_len, input_size + embedding_dim)

        _, (hidden, _) = self.rnn(x)
        out = self.fc(hidden[-1])
        return out.squeeze()

#%%
from torch.utils.data import DataLoader

dataset = SecuenciasDataset(X_num, X_canton, y)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

model = RNNConEmbeddings(input_size=X_num.shape[2], hidden_size=64, num_cantones=num_cantones, embedding_dim=8)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

for epoch in range(100):
    for x_num_batch, canton_batch, y_batch in dataloader:
        y_pred = model(x_num_batch, canton_batch)
        loss = criterion(y_pred, y_batch)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")

#%%
!jupyter nbconvert --to script 01_EDA.ipynb


#%%
