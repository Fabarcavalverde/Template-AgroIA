#%%
## MESES
#%%
import pandas as pd

class TransformadorDeFechas:
    def __init__(self, df):
        """
        Inicializa la clase con un DataFrame que contiene las columnas 'mes' y 'anio'.
        """
        self.df = df
        self.meses_es_en = {
            'enero': 'January', 'febrero': 'February', 'marzo': 'March',
            'abril': 'April', 'mayo': 'May', 'junio': 'June',
            'julio': 'July', 'agosto': 'August', 'septiembre': 'September',
            'octubre': 'October', 'noviembre': 'November', 'diciembre': 'December'
        }

    def unificar_fecha(self):
        """
        Crea una nueva columna 'fecha' combinando 'mes' y 'anio',
        luego elimina las columnas originales.
        """
        self.df['mes_en'] = self.df['mes'].str.lower().map(self.meses_es_en)
        self.df['fecha'] = pd.to_datetime(self.df['mes_en'] + ' ' + self.df['anio'].astype(str), format='%B %Y')
        self.df.drop(columns=['mes', 'anio', 'mes_en'], inplace=True)
        return self.df
