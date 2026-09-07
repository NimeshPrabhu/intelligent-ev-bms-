import pandas as pd
import numpy as np
import pickle
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam

# 1. Parse Metadata & Compute Coulomb-Counting SOH
metadata = pd.read_csv('cleaned_dataset/metadata.csv')
discharge_meta = metadata[metadata['type'].str.lower() == 'discharge'].copy()
discharge_meta['Capacity'] = pd.to_numeric(discharge_meta['Capacity'], errors='coerce')
discharge_meta = discharge_meta.dropna(subset=['Capacity']).reset_index(drop=True)
discharge_files = discharge_meta['filename'].astype(str).tolist()

def coulomb_capacity(time_s, current_a):
    time_h = time_s / 3600
    return np.trapezoid(np.abs(current_a), time_h)

base_path = "cleaned_dataset/data"
capacities = []
for fname in discharge_files:
    df = pd.read_csv(f"{base_path}/{fname}")
    cap = coulomb_capacity(df['Time'].values, df['Current_measured'].values)
    capacities.append(cap)

capacities = np.array(capacities)
SOH_cc = capacities / capacities[0]

# 2. Global Feature Scaling & Sequence Construction
SEQ_LEN = 100
FEATURES = ['Voltage_measured', 'Current_measured', 'Temperature_measured']

# Fit scaler across all raw data to capture global min/max bounds
all_raw = [pd.read_csv(f"{base_path}/{f}")[FEATURES].values for f in discharge_files]
scaler = MinMaxScaler()
scaler.fit(np.vstack(all_raw))

# Export scaler parameters for deployment
with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

def build_sequence(df, seq_len=100):
    data = scaler.transform(df[FEATURES].values)
    if len(data) >= seq_len:
        return data[:seq_len]
    else:
        pad = np.zeros((seq_len - len(data), data.shape[1]))
        return np.vstack([data, pad])

X_lstm = np.array([build_sequence(pd.read_csv(f"{base_path}/{f}"), SEQ_LEN) for f in discharge_files])
y_lstm = np.array(SOH_cc)

# 3. Train Keras Model
X_train, X_test, y_train, y_test = train_test_split(X_lstm, y_lstm, test_size=0.2, random_state=42)

model = Sequential([
    LSTM(64, return_sequences=True, input_shape=(SEQ_LEN, 3)),
    Dropout(0.2),
    LSTM(32),
    Dense(1)
])

model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])
model.fit(X_train, y_train, validation_split=0.2, epochs=50, batch_size=16, verbose=1)

# 4. Convert and Save to TFLite Format with Pure Built-in Ops
from tensorflow.keras.layers import Input

# Create an inference model with fixed batch size = 1 to allow pure built-in TFLite ops
inf_model = Sequential([
    Input(batch_shape=(1, SEQ_LEN, 3)),
    LSTM(64, return_sequences=True),
    Dropout(0.2),
    LSTM(32),
    Dense(1)
])
inf_model.set_weights(model.get_weights())

converter = tf.lite.TFLiteConverter.from_keras_model(inf_model)
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]
tflite_model = converter.convert()

with open('soc_soh_model.tflite', 'wb') as f:
    f.write(tflite_model)

# Also save keras weights for future evaluations
model.save('model.keras')

print("Export finished: Created 'soc_soh_model.tflite', 'model.keras', and 'scaler.pkl'")
