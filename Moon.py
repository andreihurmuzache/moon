import os
import MetaTrader5 as mt5
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Input, LayerNormalization, MultiHeadAttention
from tensorflow.keras.optimizers import Adam
from modules.wavelet import apply_wavelet
from modules.kalman import apply_kalman
from modules.garch import apply_garch
import logging
import matplotlib.pyplot as plt

# Configurare loguri
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Global Settings
ACCOUNT_NUMBER = 52130043
PASSWORD = "8CKzx&H9MlCEfB"
SERVER = "ICMarketsEU-Demo"
SYMBOL = "XAUUSD"
LOT_SIZE = 0.01
STOP_LOSS_PIPS = 100
TAKE_PROFIT_PIPS = 200
MODEL_PATH = "lstm_transformer_model.h5"
SCALER_PATH = "scaler.pkl"
FEATURES = ['Open', 'High', 'Low', 'Close', 'Wavelet', 'Kalman', 'GARCH']

# 1. Connect to MetaTrader 5
def connect_mt5():
    if not mt5.initialize(login=ACCOUNT_NUMBER, password=PASSWORD, server=SERVER):
        logging.error(f"Conectare eșuată la MetaTrader 5: {mt5.last_error()}")
        quit()
    logging.info("Conectat la MetaTrader 5 cu succes!")

# 2. Fetch Data
def fetch_data(symbol, timeframe, start_time, end_time):
    rates = mt5.copy_rates_range(symbol, timeframe, start_time, end_time)
    if rates is None or len(rates) == 0:
        raise ValueError("Nu s-au putut extrage datele din MetaTrader 5.")
    data = pd.DataFrame(rates)
    data['time'] = pd.to_datetime(data['time'], unit='s')
    data.rename(columns={'time': 'Date', 'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close', 'tick_volume': 'Volume'}, inplace=True)
    return data

# 3. Process Data
def process_data(data):
    data = apply_wavelet(data, column="Close")
    data['Kalman'] = apply_kalman(data['Close'])
    data['GARCH'] = apply_garch(data['Close'])

    # PCA for Dimensionality Reduction
    pca = PCA(n_components=3)
    pca_features = pca.fit_transform(data[['Open', 'High', 'Low', 'Close']])
    data[['PCA1', 'PCA2', 'PCA3']] = pca_features

    # Clustering for Market Regimes
    kmeans = KMeans(n_clusters=3, random_state=42)
    data['Cluster'] = kmeans.fit_predict(data[['Close', 'GARCH', 'Kalman']])

    return data

# 4. Train or Load Model
def train_or_load_model(data, time_steps=30, features=FEATURES):
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data[features])
    X, y = [], []
    for i in range(len(scaled_data) - time_steps):
        X.append(scaled_data[i:i + time_steps])
        y.append(scaled_data[i + time_steps, features.index('Close')])
    X, y = np.array(X), np.array(y)

    if os.path.exists(MODEL_PATH):
        logging.info(f"Modelul {MODEL_PATH} există. Îl încărcăm.")
        model = load_model(MODEL_PATH)
        return model, scaler

    # LSTM-Transformer Hybrid Model
    logging.info(f"Modelul {MODEL_PATH} nu există. Îl antrenăm.")
    lstm_transformer_model = Sequential([
        LSTM(128, return_sequences=True, input_shape=(time_steps, len(features)), activation='relu'),
        LSTM(64, activation='relu'),
        Dense(1)
    ])
    lstm_transformer_model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
    lstm_transformer_model.fit(X, y, epochs=10, batch_size=32, verbose=1)
    lstm_transformer_model.save(MODEL_PATH)
    pd.to_pickle(scaler, SCALER_PATH)
    return lstm_transformer_model, scaler

# 5. Predict and Trade
def predict_and_trade(model, scaler, data, time_steps=30, features=FEATURES):
    scaled_data = scaler.transform(data[features])
    X = [scaled_data[-time_steps:]]
    X = np.array(X)
    prediction = model.predict(X)[0][0]

    symbol_info = mt5.symbol_info_tick(SYMBOL)
    if not symbol_info:
        logging.error(f"Nu s-au găsit informații pentru simbolul {SYMBOL}.")
        return

    price = symbol_info.ask if prediction > symbol_info.bid else symbol_info.bid
    sl = price - STOP_LOSS_PIPS * 0.0001 if prediction > symbol_info.bid else price + STOP_LOSS_PIPS * 0.0001
    tp = price + TAKE_PROFIT_PIPS * 0.0001 if prediction > symbol_info.bid else price - TAKE_PROFIT_PIPS * 0.0001

    action = "buy" if prediction > symbol_info.bid else "sell"
    place_order(SYMBOL, action, LOT_SIZE, price, sl, tp)

# 6. Place Order
def place_order(symbol, action, lot_size, price, sl, tp):
    order_type = mt5.ORDER_TYPE_BUY if action == "buy" else mt5.ORDER_TYPE_SELL
    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": symbol,
        "volume": lot_size,
        "type": order_type,
        "price": price,
        "sl": sl,
        "tp": tp,
        "deviation": 10,
        "magic": 123456,
        "comment": "LSTM+Transformer",
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_IOC,
    }
    result = mt5.order_send(request)
    if result.retcode != mt5.TRADE_RETCODE_DONE:
        logging.error(f"Eroare la plasarea ordinului: {result}")
    else:
        logging.info(f"Ordin plasat cu succes: {result}")

if __name__ == "__main__":
    connect_mt5()
    start_time = datetime(2020, 1, 1)
    end_time = datetime.now()

    raw_data = fetch_data(SYMBOL, mt5.TIMEFRAME_H1, start_time, end_time)
    processed_data = process_data(raw_data)

    model, scaler = train_or_load_model(processed_data)
    predict_and_trade(model, scaler, processed_data)
