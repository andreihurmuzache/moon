import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.losses import Huber
import matplotlib.pyplot as plt
import shap
import logging
from sklearn.inspection import permutation_importance  # Import corect pentru permutation_importance
from sklearn.ensemble import RandomForestRegressor
from modules.wavelet import apply_wavelet
from modules.kalman import apply_kalman
from modules.garch import apply_garch

# Configurare loguri
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Funcție pentru curățarea datelor
def clean_data(data):
    logging.info("Verificăm existența valorilor NaN în date...")
    if data.isnull().any().any():
        logging.warning("Valorile NaN detectate în date. Le eliminăm...")
        data = data.dropna()
    logging.info(f"Datele după curățare: {data.shape}")
    return data

# Pregătim datele pentru LSTM
def prepare_lstm_data(data, time_steps, features):
    sequence_data = []
    labels = []
    for i in range(len(data) - time_steps):
        sequence_data.append(data[features].iloc[i:i + time_steps].values)
        labels.append(data['Close'].iloc[i + time_steps])
    return np.array(sequence_data), np.array(labels)

# Antrenarea modelului LSTM
def train_lstm(data, time_steps=30, features=['Close', 'ATR', 'Wavelet']):
    data = clean_data(data)

    logging.info("Scalăm datele de intrare...")
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data[features])

    X, y = [], []
    for i in range(len(scaled_data) - time_steps):
        X.append(scaled_data[i:i + time_steps])
        y.append(scaled_data[i + time_steps, 0])  # Prezicem `Close`
    X, y = np.array(X), np.array(y)

    logging.info(f"Dimensiuni X: {X.shape}, Dimensiuni y: {y.shape}")

    # Împărțire date
    split = int(0.8 * len(X))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    logging.info("Antrenăm modelul...")
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=(time_steps, len(features)), activation='relu'),
        LSTM(50, activation='relu'),
        Dense(1)
    ])

    model.compile(optimizer='adam', loss=Huber())

    history = model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=1)

    logging.info("Realizăm predicții...")
    predictions = model.predict(X_test)

    # Verificăm dacă există valori NaN în predicții
    if np.isnan(predictions).any():
        logging.error("Predicțiile conțin valori NaN!")
        raise ValueError("Predicțiile conțin valori NaN!")

    mse = mean_squared_error(y_test, predictions)
    logging.info(f"Mean Squared Error: {mse}")

    # Vizualizare predicții
    plt.figure(figsize=(12, 6))
    plt.plot(range(len(y_test)), y_test, label="Real Values", linestyle='-', color='blue')
    plt.plot(range(len(predictions)), predictions, label="Predictions", linestyle='--', color='orange')
    plt.legend()
    plt.title("LSTM Predictions vs Real Values")
    plt.xlabel("Time Steps")
    plt.ylabel("Scaled Close Prices")
    plt.grid()
    plt.show()

    # Permutation Importance
    logging.info("Calculăm Permutation Importance...")
    X_test_flat = X_test[:, -1, :]  # Luăm ultima observație din fiecare secvență
    surrogate_model = RandomForestRegressor()
    surrogate_model.fit(X_test_flat, y_test)
    perm_importance = permutation_importance(surrogate_model, X_test_flat, y_test, n_repeats=10, random_state=42)
    
    plt.bar(features, perm_importance.importances_mean)
    plt.title("Permutation Importance")
    plt.xlabel("Features")
    plt.ylabel("Importance")
    plt.show()

    # SHAP explainer
    logging.info("Generăm explicațiile SHAP...")
    explainer = shap.Explainer(surrogate_model, X_test_flat)
    shap_values = explainer(X_test_flat)

    plt.title("SHAP Summary Plot")
    shap.summary_plot(shap_values, features=features, show=True)

    return model

if __name__ == "__main__":
    # Încarcă datele procesate
    input_csv = "processed_data.csv"

    try:
        data = pd.read_csv(input_csv)
        logging.info(f"Fișierul procesat {input_csv} încărcat cu succes.")
    except FileNotFoundError:
        logging.error(f"Fișierul {input_csv} nu a fost găsit.")
        raise

    # Specificăm caracteristicile
    time_steps = 30
    features = ['Open', 'High', 'Low', 'Close', 'Volume', 'ATR', 'Wavelet']

    # Antrenăm modelul și generăm grafice
    train_lstm(data, time_steps=time_steps, features=features)
