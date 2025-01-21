import MetaTrader5 as mt5
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import pickle  # Pentru salvare modele
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
import ESN  # Importăm clasa ESN

# 1. Conectare la MetaTrader 5
login = 52130043
password = "8CKzx&H9MlCEfB"
server = "ICMarketsEU-Demo"

if not mt5.initialize(login=login, password=password, server=server):
    print("Conectare eșuată la MT5!", mt5.last_error())
    quit()
print("Conectat la MT5 cu succes!")

# 2. Extragere date
def fetch_max_data(symbol, timeframe):
    rates = []
    start_time = datetime(2016, 1, 1)
    end_time = datetime(2025, 1, 1)

    while start_time < end_time:
        batch = mt5.copy_rates_range(symbol, timeframe, start_time, end_time)
        if batch is not None and len(batch) > 0:
            rates.extend(batch)
            start_time = datetime.fromtimestamp(batch[-1]['time']) + timedelta(seconds=1)
        else:
            break
    return rates

# 3. Procesare date brute
def process_raw_data(raw_data):
    # Verificăm dacă datele brute sunt goale
    if len(raw_data) == 0:
        raise ValueError("Datele brute sunt goale.")

    # Convertim fiecare element într-un tuplu, dacă este nevoie
    if isinstance(raw_data[0], np.void):
        raw_data = [tuple(row.tolist()) for row in raw_data]

    # Verificăm structura după conversie
    if not isinstance(raw_data[0], (tuple, list)):
        raise ValueError(f"Structura datelor brute este neașteptată: {type(raw_data[0])}.")

    # Convertim în DataFrame
    processed = pd.DataFrame(raw_data, columns=['time', 'open', 'high', 'low', 'close', 'tick_volume', 'spread', 'real_volume'])
    processed['time'] = pd.to_datetime(processed['time'], unit='s')
    processed.rename(columns={'time': 'Date', 'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close', 'tick_volume': 'Volume'}, inplace=True)

    return processed[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']]

# 4. Antrenare model ESN
def train_esn(data):
    X_train = data['Close'].values[:-100].reshape(-1, 1)
    y_train = data['Close'].values[1:-99]
    esn = ESN.EchoStateNetwork(n_reservoir=500, spectral_radius=1.2, sparsity=0.1, noise=0.001)
    esn.fit(X_train, y_train)
    with open("esn_model.pkl", "wb") as f:
        pickle.dump(esn, f)
    print("Model ESN salvat ca 'esn_model.pkl'.")

# 5. Antrenare model LSTM
def train_lstm(data):
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data['Close'].values.reshape(-1, 1))

    X, y = [], []
    for i in range(50, len(scaled_data)):
        X.append(scaled_data[i-50:i, 0])
        y.append(scaled_data[i, 0])
    X, y = np.array(X), np.array(y)
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))

    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=(X.shape[1], 1)),
        Dropout(0.2),
        LSTM(25, return_sequences=False),
        Dropout(0.2),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X, y, epochs=10, batch_size=32, verbose=1)
    model.save("lstm_model.h5")
    print("Model LSTM salvat ca 'lstm_model.h5'.")

# 6. Executare completă
symbol = "USDJPY"
timeframe = mt5.TIMEFRAME_H1

print(f"Fetching maximum data for {symbol}...")
raw_data = fetch_max_data(symbol, timeframe)

if raw_data:
    structured_data = process_raw_data(raw_data)
    print("Antrenare model ESN...")
    train_esn(structured_data)
    print("Antrenare model LSTM...")
    train_lstm(structured_data)
else:
    print("Nu s-au găsit date pentru acest simbol.")

mt5.shutdown()
print("MT5 deconectat.")
