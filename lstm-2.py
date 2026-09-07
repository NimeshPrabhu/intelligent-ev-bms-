import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

def get_soh_labels(data_folder, discharge_files):
    capacities = []
    for fname in discharge_files:
        df = pd.read_csv(os.path.join(data_folder, fname))
        time_h = df['Time'].values / 3600
        cap = np.trapezoid(np.abs(df['Current_measured'].values), time_h)
        capacities.append(cap)
    capacities = np.array(capacities)
    return capacities / capacities[0]

def load_and_pad_sequence(file_path, features, seq_len):
    df = pd.read_csv(file_path)
    data = df[features].values
    if len(data) >= seq_len:
        return data[:seq_len]
    padding = np.zeros((seq_len - len(data), data.shape[1]))
    return np.vstack([data, padding])

def get_preprocessed_data(discharge_files, base_path, soh_labels, seq_len=100):
    features = ['Voltage_measured', 'Current_measured', 'Temperature_measured']
    X_list = [load_and_pad_sequence(os.path.join(base_path, f), features, seq_len) for f in discharge_files]
    X = np.array(X_list)
    y = np.array(soh_labels)
    N, L, F = X.shape
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X.reshape(-1, F)).reshape(N, L, F)
    return X_scaled, y

def build_lstm_model(input_shape):
    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=input_shape),
        Dropout(0.2),
        LSTM(32),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

if __name__ == "__main__":
    metadata_path = "cleaned_dataset/metadata.csv"
    data_folder = "cleaned_dataset/data"

    metadata = pd.read_csv(metadata_path)
    discharge_meta = metadata[metadata['type'].str.lower() == 'discharge']
    discharge_files = discharge_meta['filename'].astype(str).tolist()

    SOH_cc = get_soh_labels(data_folder, discharge_files)
    X, y = get_preprocessed_data(discharge_files, data_folder, SOH_cc)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = build_lstm_model((X.shape[1], X.shape[2]))
    history = model.fit(X_train, y_train, validation_split=0.2, epochs=50, batch_size=16, verbose=1)

    y_all_pred = np.clip(model.predict(X).flatten(), 0, 1.05)

    plt.figure(figsize=(10, 5))
    plt.plot(y, label='Actual SOH', color='black')
    plt.plot(y_all_pred, label='Predicted SOH', linestyle='--', color='red')
    plt.title("Battery Health Prediction")
    plt.xlabel("Cycle Index")
    plt.ylabel("SOH")
    plt.legend()
    plt.show()
