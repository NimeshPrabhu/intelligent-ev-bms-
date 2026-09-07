"""
TFLite LSTM Battery SOH Model Evaluation Script
Computes accuracy metrics (MAE, RMSE, R^2, Max Error, MAPE) against ground truth
and exports a visual validation comparison chart.
"""

import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Try tflite_runtime first, then tensorflow
try:
    import tflite_runtime.interpreter as tflite
except ImportError:
    try:
        import tensorflow.lite as tflite
    except ImportError:
        raise ImportError("Please install tflite-runtime or tensorflow to evaluate the model.")

SEQ_LEN = 100
FEATURES = ['Voltage_measured', 'Current_measured', 'Temperature_measured']
DATA_FOLDER = "cleaned_dataset/data"
METADATA_PATH = "cleaned_dataset/metadata.csv"
MODEL_PATH = "soc_soh_model.tflite"
SCALER_PATH = "scaler.pkl"
OUTPUT_PLOT = "tflite_evaluation_metrics.png"

def coulomb_capacity(time_s, current_a):
    """Integrates current over time (Ah) via trapezoidal rule."""
    time_h = time_s / 3600.0
    return np.trapezoid(np.abs(current_a), time_h)

def load_dataset():
    """Loads discharge cycles, computes Coulomb counting SOH, and formats sequences."""
    print("[1/4] Loading metadata and dataset...")
    metadata = pd.read_csv(METADATA_PATH)
    discharge_meta = metadata[metadata['type'].str.lower() == 'discharge'].copy()
    discharge_meta['Capacity'] = pd.to_numeric(discharge_meta['Capacity'], errors='coerce')
    discharge_meta = discharge_meta.dropna(subset=['Capacity']).reset_index(drop=True)
    discharge_files = discharge_meta['filename'].astype(str).tolist()

    # Compute Coulomb counting capacity for each file
    capacities = []
    for fname in discharge_files:
        fpath = os.path.join(DATA_FOLDER, fname)
        df = pd.read_csv(fpath)
        cap = coulomb_capacity(df['Time'].values, df['Current_measured'].values)
        capacities.append(cap)

    capacities = np.array(capacities)
    # Ground truth SOH: normalized to first cycle capacity
    true_soh = capacities / capacities[0]

    # Load scaler
    with open(SCALER_PATH, "rb") as f:
        scaler = pickle.load(f)

    # Build sequence data: (N, 100, 3)
    X_sequences = []
    for fname in discharge_files:
        fpath = os.path.join(DATA_FOLDER, fname)
        df = pd.read_csv(fpath)
        data = scaler.transform(df[FEATURES].values)
        if len(data) >= SEQ_LEN:
            seq = data[:SEQ_LEN]
        else:
            pad = np.zeros((SEQ_LEN - len(data), data.shape[1]))
            seq = np.vstack([data, pad])
        X_sequences.append(seq)

    X_sequences = np.array(X_sequences, dtype=np.float32)
    return X_sequences, true_soh, discharge_files

def run_tflite_inference(X_sequences):
    """Runs batch inference using the TFLite Interpreter."""
    print(f"[2/4] Initializing TFLite interpreter from '{MODEL_PATH}'...")
    interpreter = tflite.Interpreter(model_path=MODEL_PATH)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    input_index = input_details[0]['index']
    output_index = output_details[0]['index']

    print(f"[3/4] Running inference over {len(X_sequences)} cycles...")
    y_pred = []
    for i in range(len(X_sequences)):
        sample = np.expand_dims(X_sequences[i], axis=0).astype(np.float32)
        interpreter.set_tensor(input_index, sample)
        interpreter.invoke()
        output = interpreter.get_tensor(output_index)
        y_pred.append(float(output[0][0]))

    y_pred = np.array(y_pred)
    return y_pred

def compute_and_display_metrics(y_true, y_pred):
    """Computes standard evaluation metrics for regression."""
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    max_err = np.max(np.abs(y_true - y_pred))
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100.0

    print("\n" + "=" * 55)
    print("        TFLITE MODEL EVALUATION RESULTS (SOH)")
    print("=" * 55)
    print(f" Total Evaluation Cycles:  {len(y_true)}")
    print(f" Mean Absolute Error (MAE):     {mae:.4f}  ({mae * 100:.2f}%)")
    print(f" Root Mean Squared Error (RMSE): {rmse:.4f}  ({rmse * 100:.2f}%)")
    print(f" R² Score (Variance Explained): {r2:.4f}")
    print(f" Mean Abs Percentage Error (MAPE): {mape:.2f}%")
    print(f" Maximum Absolute Error:        {max_err:.4f}  ({max_err * 100:.2f}%)")
    print("=" * 55 + "\n")

    return {
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
        "mape": mape,
        "max_error": max_err
    }

def plot_results(y_true, y_pred):
    """Generates a 3-panel validation figure."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')

    # Panel 1: Degradation Trend Over Cycles
    axes[0].plot(y_true * 100, label='True SOH (Coulomb Counting)', color='#2563eb', linewidth=2.2)
    axes[0].plot(y_pred * 100, label='TFLite Predicted SOH', color='#dc2626', linestyle='--', linewidth=2.0)
    axes[0].set_title('SOH Degradation Profile Over Cycles', fontsize=12, fontweight='bold')
    axes[0].set_xlabel('Discharge Cycle Index', fontsize=10)
    axes[0].set_ylabel('State of Health (%)', fontsize=10)
    axes[0].legend(loc='upper right')
    axes[0].grid(True, alpha=0.3)

    # Panel 2: Predicted vs Actual Scatter Plot
    axes[1].scatter(y_true * 100, y_pred * 100, color='#0d9488', alpha=0.6, edgecolors='k', linewidth=0.5)
    min_val = min(np.min(y_true * 100), np.min(y_pred * 100)) - 2
    max_val = max(np.max(y_true * 100), np.max(y_pred * 100)) + 2
    axes[1].plot([min_val, max_val], [min_val, max_val], 'r--', label='Ideal 1:1 Line', linewidth=1.8)
    axes[1].set_title('True SOH vs. Predicted SOH', fontsize=12, fontweight='bold')
    axes[1].set_xlabel('Actual SOH (%)', fontsize=10)
    axes[1].set_ylabel('Predicted SOH (%)', fontsize=10)
    axes[1].legend(loc='upper left')
    axes[1].grid(True, alpha=0.3)

    # Panel 3: Error Residual Distribution
    residuals = (y_pred - y_true) * 100
    axes[2].hist(residuals, bins=25, color='#8b5cf6', edgecolor='black', alpha=0.75)
    axes[2].axvline(0, color='red', linestyle='--', linewidth=1.5)
    axes[2].set_title('Prediction Error Distribution (Residuals)', fontsize=12, fontweight='bold')
    axes[2].set_xlabel('Error (Predicted SOH - True SOH %)', fontsize=10)
    axes[2].set_ylabel('Frequency', fontsize=10)
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUTPUT_PLOT, dpi=200)
    print(f"[4/4] Saved evaluation figure to '{OUTPUT_PLOT}'")

if __name__ == "__main__":
    X, y_true, files = load_dataset()
    y_pred = run_tflite_inference(X)
    compute_and_display_metrics(y_true, y_pred)
    plot_results(y_true, y_pred)
