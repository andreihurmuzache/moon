import os
import MetaTrader5 as mt5
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense
from sklearn.metrics import mean_squared_error
from modules.wavelet import apply_wavelet
from modules.kalman import apply_kalman
from modules.garch import apply_garch
from modules.kmeans import apply_kmeans
import logging
import matplotlib.pyplot as plt

# Configurare loguri
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# 1. Funcție pentru extragerea și procesarea datelor brute
def fetch_and_process_data(symbol, timeframe, start_time, end_time, output_csv="raw_data.csv"):
    if os.path.exists(output_csv):
        logging.info(f"Fișierul {output_csv} există deja. Îl încărcăm.")
        return pd.read_csv(output_csv)

    if not mt5.initialize():
        logging.error("Conectare eșuată la MetaTrader 5!")
        quit()

    logging.info("Conectat la MetaTrader 5 cu succes.")
    rates = mt5.copy_rates_range(symbol, timeframe, start_time, end_time)

    if rates is None or len(rates) == 0:
        logging.error(f"Nu s-au găsit date pentru simbolul {symbol}.")
        quit()

    raw_data = pd.DataFrame(rates)
    raw_data['time'] = pd.to_datetime(raw_data['time'], unit='s')
    raw_data.rename(columns={
        'time': 'Date',
        'open': 'Open',
        'high': 'High',
        'low': 'Low',
        'close': 'Close',
        'tick_volume': 'Volume'
    }, inplace=True)

    # Salvăm datele brute
    raw_data.to_csv(output_csv, index=False)
    logging.info(f"Datele brute au fost salvate în {output_csv}.")
    return raw_data

# 2. Funcție pentru analiza datelor
def analyze_data(data):
    data = apply_wavelet(data, column="Close")
    data['Kalman'] = apply_kalman(data['Close'])
    data['GARCH'] = apply_garch(data['Close'])
    data, _ = apply_kmeans(data, n_clusters=3)
    return data

# 3. Funcție pentru antrenarea sau încărcarea modelului LSTM
from tensorflow.keras.losses import MeanSquaredError

def train_or_load_model(data, model_path="lstm_model.h5", time_steps=30, features=['Open', 'High', 'Low', 'Close', 'Wavelet']):
    scaler = MinMaxScaler(feature_range=(0, 1))
    data = data.dropna(subset=features)
    scaled_data = scaler.fit_transform(data[features])

    X, y = [], []
    for i in range(len(scaled_data) - time_steps):
        X.append(scaled_data[i:i + time_steps])
        y.append(scaled_data[i + time_steps, 3])  # Prezicem 'Close'
    X, y = np.array(X), np.array(y)

    if os.path.exists(model_path):
        logging.info(f"Modelul {model_path} există. Îl încărcăm.")
        try:
            model = load_model(model_path)
        except Exception as e:
            logging.error(f"Eroare la încărcarea modelului: {e}. Reantrenăm modelul.")
            model = None
    else:
        logging.info(f"Modelul {model_path} nu există. Îl antrenăm.")
        model = None

    if model is None:
        model = Sequential([
            LSTM(128, return_sequences=True, input_shape=(time_steps, len(features)), activation='relu'),
            LSTM(64, activation='relu'),
            Dense(1)
        ])
        model.compile(optimizer='adam', loss=MeanSquaredError())
        model.fit(X, y, epochs=10, batch_size=32, verbose=1)
        model.save(model_path)

    return model, scaler


# 4. Funcție pentru evaluarea pe out-of-sample
def evaluate_out_of_sample(data, model, scaler, time_steps=30, features=['Open', 'High', 'Low', 'Close', 'Wavelet'], output_csv="out_of_sample_predictions.csv"):
    scaled_data = scaler.transform(data[features])
    X, y = [], []
    for i in range(len(scaled_data) - time_steps):
        X.append(scaled_data[i:i + time_steps])
        y.append(scaled_data[i + time_steps, 3])  # Prezicem 'Close'
    X, y = np.array(X), np.array(y)

    predictions = model.predict(X)
    mse = mean_squared_error(y, predictions)
    logging.info(f"Mean Squared Error (Out-of-Sample): {mse}")

    # Adăugăm predicțiile la DataFrame
    predictions_full = [np.nan] * time_steps + predictions.flatten().tolist()
    data['Predictions'] = predictions_full

    data.to_csv(output_csv, index=False)
    logging.info(f"Predicțiile pentru datele out-of-sample au fost salvate în {output_csv}.")

    # Plot
    plt.figure(figsize=(12, 6))
    plt.plot(range(len(y)), y, label="Valori reale", linestyle='-', color='blue')
    plt.plot(range(len(predictions)), predictions, label="Predicții", linestyle='--', color='orange')
    plt.title("LSTM Predictions vs Real Values (Out-of-Sample)")
    plt.xlabel("Time Steps")
    plt.ylabel("Close Price")
    plt.legend()
    plt.grid()
    plt.show()

# 5. Funcție principală
if __name__ == "__main__":
    symbol = "XAUUSD"
    timeframe = mt5.TIMEFRAME_H1
    start_time = datetime(2020, 1, 1)
    end_time = datetime(2023, 12, 31)  # Date de antrenament
    out_of_sample_start = datetime(2024, 1, 1)
    out_of_sample_end = datetime(2025, 1, 1)

    # Datele de antrenament
    raw_data_file = "train_data.csv"
    train_data = fetch_and_process_data(symbol, timeframe, start_time, end_time, output_csv=raw_data_file)
    train_data = analyze_data(train_data)

    # Antrenare sau încărcare model
    lstm_model, lstm_scaler = train_or_load_model(train_data)

    # Datele out-of-sample
    out_of_sample_file = "out_of_sample_data.csv"
    out_of_sample_data = fetch_and_process_data(symbol, timeframe, out_of_sample_start, out_of_sample_end, output_csv=out_of_sample_file)
    out_of_sample_data = analyze_data(out_of_sample_data)

    # Evaluare out-of-sample
    evaluate_out_of_sample(out_of_sample_data, lstm_model, lstm_scaler)
