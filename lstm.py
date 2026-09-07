import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam

### Data Preprocessing Functions

def preprocess_battery_data(discharge_files, base_path, soh_labels, seq_len=100):
    """
    Loads, scales, and sequences the data.
    """
    features = ['Voltage_measured', 'Current_measured', 'Temperature_measured']
    X_list = []

    # Global scaler to maintain relative values between healthy/unhealthy cycles
    scaler = MinMaxScaler()

    for fname in discharge_files:
        df = pd.read_csv(f"{base_path}/{fname}")
        data = df[features].values

        # Consistent sequencing with padding
        if len(data) >= seq_len:
            seq = data[:seq_len]
        else:
            pad = np.zeros((seq_len - len(data), data.shape[1]))
            seq = np.vstack([data, pad])
        X_list.append(seq)

    X = np.array(X_list)
    y = np.array(soh_labels)

    # Reshape to 2D for scaling, then back to 3D for LSTM
    N, L, F = X.shape
    X_reshaped = X.reshape(-1, F)
    X_scaled = scaler.fit_transform(X_reshaped).reshape(N, L, F)

    return X_scaled, y

### Model Architecture

def build_lstm_model(input_shape):
    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=input_shape),
        Dropout(0.2),
        LSTM(32),
        Dense(1)
    ])
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])
    return model

def plot_performance(history, y_test, y_pred, y_true_all, y_pred_all):
    fig, ax = plt.subplots(1, 3, figsize=(18, 5))

    # Loss Curve
    ax[0].plot(history.history['loss'], label='Train')
    ax[0].plot(history.history['val_loss'], label='Val')
    ax[0].set_title("Model Loss (MSE)")
    ax[0].set_xlabel("Epoch")
    ax[0].legend()

    # Regression Plot
    ax[1].scatter(y_test, y_pred, alpha=0.5)
    ax[1].plot([0, 1.1], [0, 1.1], '--r')
    ax[1].set_title("Predicted vs True SOH")
    ax[1].set_xlabel("Actual SOH")
    ax[1].set_ylabel("Predicted SOH")

    # Comparison Trend
    ax[2].plot(y_true_all, label='Actual (Coulomb Counting)', alpha=0.7)
    ax[2].plot(y_pred_all, label='LSTM Predicted', alpha=0.7)
    ax[2].set_title("SOH Degradation Trend")
    ax[2].set_xlabel("Cycle Index")
    ax[2].legend()

    plt.tight_layout()
    plt.show()

### Main

# Prepare Data
X, y = preprocess_battery_data(discharge_files, "cleaned_dataset/data", SOH_cc)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Model
model = build_lstm_model((100, 3))
history = model.fit(X_train, y_train, validation_split=0.2, epochs=50, batch_size=16, verbose=0)

# Evaluate and Predict
y_test_pred = model.predict(X_test).flatten()
y_all_pred = np.clip(model.predict(X).flatten(), 0, 1.05)

# Visualize Results
plot_performance(history, y_test, y_test_pred, y, y_all_pred)
