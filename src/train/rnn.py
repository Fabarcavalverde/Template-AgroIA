"""
Clase: rnn

Objetivo: Clase para cargar modelo RNN entrenado y predecir secuencias temporales

Cambios:
    1. Creación de clase - Fiorella, 14-07-2025
"""

import os
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from typing import Dict, Any

class rnn:
    def __init__(self, ruta_raiz: str):
        self.model = load_model(os.path.join(ruta_raiz, 'models/modelo_RNN_Papas.h5'))
        self.window_size = 30  #