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

# Configurare loguri
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Date de conectare (înlocuiește cu datele tale reale)
ACCOUNT_NUMBER = 52130043
PASSWORD = "8CKzx&H9MlCEfB"
SERVER = "ICMarketsEU-Demo"
SYMBOL = "XAUUSD"
LOT_SIZE = 0.01
STOP_LOSS = 100  # Pips
TAKE_PROFIT = 200  # Pips

# Funcție pentru conectare la MetaTrader 5
def connect_mt5():
    if not mt5.initialize(login=ACCOUNT_NUMBER, password=PASSWORD, server=SERVER):
        logging.error(f"Conectare eșuată la MetaTrader 5: {mt5.last_error()}")
        quit()
    logging.info("Conectat la MetaTrader 5 cu succes!")

# Funcție pentru extragerea datelor
def fetch_data(symbol, timeframe, start_time, end_time):
    rates = mt5.copy_rates_range(symbol, timeframe, start_time, end_time)
    if rates is None or len(rates) == 0:
        raise ValueError("Nu s-au putut extrage datele din MetaTrader 5.")
    data = pd.DataFrame(rates)
    data['time'] = pd.to_datetime(data['time'], unit='s')
    data.rename(columns={'time': 'Date', 'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close', 'tick_volume': 'Volume'}, inplace=True)
    return data

# Funcție pentru procesarea datelor
def process_data(data):
    data = apply_wavelet(data, column="Close")
    data['Kalman'] = apply_kalman(data['Close'])
    data['GARCH'] = apply_garch(data['Close'])
    data, _ = apply_kmeans(data, n_clusters=3)
    return data

# Funcție pentru antrenarea/incarcarea modelului
def train_or_load_model(data, time_steps=30, features=['Open', 'High', 'Low', 'Close', 'Wavelet'], model_path="lstm_model.h5"):
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data[features])

    X, y = [], []
    for i in range(len(scaled_data) - time_steps):
        X.append(scaled_data[i:i + time_steps])
        y.append(scaled_data[i + time_steps, features.index('Close')])
    X, y = np.array(X), np.array(y)

    if os.path.exists(model_path):
        logging.info(f"Modelul {model_path} există. Îl încărcăm.")
        model = load_model(model_path)
        return model, scaler

    logging.info(f"Modelul {model_path} nu există. Îl antrenăm.")
    model = Sequential([
        LSTM(128, return_sequences=True, input_shape=(time_steps, len(features)), activation='relu'),
        LSTM(64, activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    model.fit(X, y, epochs=10, batch_size=32, verbose=1)
    model.save(model_path)
    pd.to_pickle(scaler, "scaler.pkl")
    return model, scaler

# Funcție pentru predicții și trading live
def predict_and_trade(model, scaler, data, time_steps=30, features=['Open', 'High', 'Low', 'Close', 'Wavelet']):
    # Filtrăm doar caracteristicile relevante
    try:
        data = data[features]
    except KeyError as e:
        raise ValueError(f"Eroare: Lipsesc coloanele necesare pentru predicții. {e}")

    # Scalăm datele folosind același scaler ca în timpul antrenării
    scaled_data = scaler.transform(data)

    # Construim setul de date pentru predicție
    X = [scaled_data[-time_steps:]]  # Ultimele 'time_steps' observații
    X = np.array(X)

    # Realizăm predicția
    prediction = model.predict(X)[0][0]

    # Obținem informațiile despre simbol
    symbol_info = mt5.symbol_info_tick(SYMBOL)
    if not symbol_info:
        logging.error(f"Nu s-au găsit informații pentru simbolul {SYMBOL}.")
        return

    # Determinăm prețul de intrare și nivelurile SL/TP
    price = symbol_info.ask if prediction > symbol_info.bid else symbol_info.bid
    sl = price - STOP_LOSS * 0.0001 if prediction > symbol_info.bid else price + STOP_LOSS * 0.0001
    tp = price + TAKE_PROFIT * 0.0001 if prediction > symbol_info.bid else price - TAKE_PROFIT * 0.0001

    action = "buy" if prediction > symbol_info.bid else "sell"
    place_order(SYMBOL, action, LOT_SIZE, price, sl, tp)


# Funcție pentru plasarea ordinului
# Funcție pentru plasarea ordinului
def place_order(symbol, action, lot_size, price, sl, tp):
    order_type = mt5.ORDER_TYPE_BUY if action == "buy" else mt5.ORDER_TYPE_SELL

    # Verificăm dacă simbolul este activ
    symbol_info = mt5.symbol_info(symbol)
    if symbol_info is None:
        logging.error(f"Simbolul {symbol} nu este disponibil pe platformă.")
        return
    
    if not symbol_info.visible:
        logging.error(f"Simbolul {symbol} nu este activ. Încercăm să-l activăm...")
        if not mt5.symbol_select(symbol, True):
            logging.error(f"Nu am reușit să activăm simbolul {symbol}.")
            return
        else:
            logging.info(f"Simbolul {symbol} a fost activat.")

    # Obținem tick-ul curent
    tick = mt5.symbol_info_tick(symbol)
    if tick is None:
        logging.error(f"Nu am reușit să obținem tick-ul pentru simbolul {symbol}.")
        return

    # Construim cererea de tranzacționare
    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": symbol,
        "volume": lot_size,
        "type": order_type,
        "price": tick.ask if action == "buy" else tick.bid,
       # "sl": sl,
        #"tp": tp,
        "deviation": 10,
        "magic": 123456,
        "comment": "LSTM prediction",
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_IOC,  # FOK sau IOC, depinde de broker
    }

    logging.info(f"Trimitem cererea: {request}")
    result = mt5.order_send(request)

    # Logăm răspunsul serverului
    # Verificăm și logăm toate informațiile
    if result.retcode != mt5.TRADE_RETCODE_DONE:
        logging.error(f"Eroare la plasarea ordinului: {result}")
        logging.error(f"Cod de retur: {result.retcode}")
        logging.error(f"Comentariu broker: {result.comment}")
        logging.error(f"Detalii complete: {result}")
    else:
        logging.info(f"Ordin plasat cu succes: {result}")
        logging.info(f"Cod de retur: {result.retcode}")
        logging.info(f"Detalii complete: {result}")
    
 # Returnăm codul de retur pentru analiză ulterioară
    return result.retcode, result

if __name__ == "__main__":
    connect_mt5()

    start_time = datetime(2020, 1, 1)
    end_time = datetime.now()

    data = fetch_data(SYMBOL, mt5.TIMEFRAME_H1, start_time, end_time)
    processed_data = process_data(data)

    model, scaler = train_or_load_model(processed_data)
    predict_and_trade(model, scaler, processed_data)
    
