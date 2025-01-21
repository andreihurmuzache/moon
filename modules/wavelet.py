import numpy as np
from scipy.signal import cwt, ricker
import pandas as pd  # Asigură-te că pandas este importat

def apply_wavelet(data, column="Close", width=10):
    if not isinstance(data, pd.DataFrame):
        raise ValueError("Datele trebuie să fie un pandas.DataFrame.")
    
    if column not in data.columns:
        raise ValueError(f"Coloana '{column}' nu există în DataFrame.")
    
    wavelet_result = cwt(data[column], ricker, np.arange(1, width + 1))
    data['Wavelet'] = wavelet_result.mean(axis=0)  # Media valorilor wavelet
    return data
