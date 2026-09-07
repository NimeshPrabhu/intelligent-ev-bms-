import pandas as pd

metadata = pd.read_csv('cleaned_dataset/metadata.csv')
metadata.head()

# Extract the discharge files from the data as per given in the meta data.
discharge_meta = metadata[
    metadata['type'].str.lower() == 'discharge'
]

discharge_files = discharge_meta['filename'].astype(str).tolist()

discharge_files[:5]

# taking out the capacity column and null values
discharge_meta['Capacity'].isna().sum(), len(discharge_meta)

discharge_meta = discharge_meta.copy()

discharge_meta['Capacity'] = pd.to_numeric(
    discharge_meta['Capacity'],
    errors='coerce'
)

discharge_meta = discharge_meta.dropna(subset=['Capacity'])
discharge_meta = discharge_meta.reset_index(drop=True)

#### SOH ####
import numpy as np

capacity = discharge_meta['Capacity'].values
SOH = capacity / capacity[0]

#### Plot for global trend ####
import matplotlib.pyplot as plt

plt.figure(figsize=(8,4))
plt.plot(SOH, linewidth=2)
plt.xlabel("Discharge Cycle Index")
plt.ylabel("SOH")
plt.title("Battery State of Health (SOH)")
plt.grid(True)
plt.show()

# Plot for one battery
battery_id = discharge_meta['battery_id'].iloc[0]
battery_df = discharge_meta[
    discharge_meta['battery_id'] == battery_id
].copy()

battery_df = battery_df.sort_values('start_time').reset_index(drop=True)

capacity = battery_df['Capacity'].values
SOH = capacity / capacity[0]

# Plot for degradation of one battery
plt.figure(figsize=(6,4))
plt.plot(SOH, marker='o')
plt.xlabel("Discharge Cycle")
plt.ylabel("SOH")
plt.title(f"SOH Degradation - Battery {battery_id}")
plt.grid(True)
plt.show()


####################################################

file_path = 'cleaned_dataset/data/00001.csv'
df = pd.read_csv(file_path)

def coulomb_capacity(time_s, current_a):
    time_h = time_s / 3600
    return np.trapezoid(np.abs(current_a), time_h)

capacity_i = coulomb_capacity(
    df['Time'].values,
    df['Current_measured'].values
)

capacity_0 = capacity_i
SOH_i = capacity_i / capacity_0

base_path = "cleaned_dataset/data"

capacities = []

for fname in discharge_files:
    file_path = f"{base_path}/{fname}"
    df = pd.read_csv(file_path)
    cap = coulomb_capacity(
        df['Time'].values,
        df['Current_measured'].values
    )
    capacities.append(cap)

capacities = np.array(capacities)
SOH_cc = capacities / capacities[0]

#### Coloumb-Counting SOH ####
plt.figure(figsize=(8,4))
plt.plot(SOH_cc, marker='o')
plt.xlabel("Discharge Cycle Index")
plt.ylabel("SOH")
plt.title("SOH via Coulomb Counting")
plt.grid(True)
plt.show()

#########################################################

SEQ_LEN = 100
FEATURES = ['Voltage_measured', 'Current_measured', 'Temperature_measured']

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()

def build_sequence(df, seq_len=100):
    data = df[FEATURES].values
    # Normalize per cycle
    data = scaler.fit_transform(data)
    if len(data) >= seq_len:
        return data[:seq_len]
    else:
        pad = np.zeros((seq_len - len(data), data.shape[1]))
        return np.vstack([data, pad])


base_path = "cleaned_dataset/data"

X_lstm = []
y_lstm = []

for i, fname in enumerate(discharge_files):
    df = pd.read_csv(f"{base_path}/{fname}")
    seq = build_sequence(df, SEQ_LEN)
    X_lstm.append(seq)
    y_lstm.append(SOH_cc[i])  # Coulomb-counting SOH

X_lstm = np.array(X_lstm)
y_lstm = np.array(y_lstm)

# X_lstm.shape, y_lstm.shape

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X_lstm, y_lstm, test_size=0.2, random_state=42
)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam

model = Sequential([
    LSTM(64, return_sequences=True, input_shape=(SEQ_LEN, 3)),
    Dropout(0.2),
    LSTM(32),
    Dense(1)
])

model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='mse',
    metrics=['mae']
)

model.summary()

history = model.fit(
    X_train, y_train,
    validation_split=0.2,
    epochs=50,
    batch_size=16,
    verbose=1
)

from sklearn.metrics import mean_absolute_error

y_pred = model.predict(X_test).flatten()

mae = mean_absolute_error(y_test, y_pred)

plt.figure(figsize=(6,5))
plt.scatter(y_test, y_pred)
plt.plot([0,1], [0,1], '--r')
plt.xlabel("True SOH")
plt.ylabel("Predicted SOH")
plt.title("LSTM-based SOH Estimation")
plt.grid(True)
plt.show()

plt.figure(figsize=(6,4))
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.xlabel("Epoch")
plt.ylabel("MSE")
plt.legend()
plt.grid(True)
plt.show()

y_pred_lstm = model.predict(X_lstm).flatten()
y_pred_lstm = np.clip(y_pred_lstm, 0, 1.05)

min_len = min(
    len(SOH_cc),
    len(y_pred_lstm)
)

plt.figure(figsize=(10,5))
plt.plot(SOH_cc[:min_len], label='Coulomb Counting')
plt.plot(y_pred_lstm[:min_len], label='LSTM-based')

plt.xlabel("Cycle Index")
plt.ylabel("SOH")
plt.title("Comparison of SOH Estimation Methods")
plt.legend()
plt.grid(True)
plt.show()
