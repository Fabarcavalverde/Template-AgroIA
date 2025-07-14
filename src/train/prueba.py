import os
from src.train.rnn import rnn
from src.utils.metrics import obtener_ruta_app


base_path = obtener_ruta_app("TemplateAgroIA")
modelo_rnn = rnn(base_path)


ruta_archivo = os.path.join(base_path, "data", "raw", "dataset_prueba_rnn.csv")


prediccion = modelo_rnn.predecir_archivo(ruta_archivo)
