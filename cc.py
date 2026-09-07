import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

def coulomb_capacity(time_s, current_a):
    """
    Compute capacity using Coulomb counting.
    """
    time_h = time_s / 3600
    return np.trapezoid(np.abs(current_a), time_h)

def plot_global_soh(metadata_path):
    """
    Plots SOH trend across all batteries.
    """
    metadata = pd.read_csv(metadata_path)

    discharge_meta = metadata[
        metadata['type'].str.lower() == 'discharge'
    ].copy()

    discharge_meta['Capacity'] = pd.to_numeric(
        discharge_meta['Capacity'],
        errors='coerce'
    )

    discharge_meta = discharge_meta.dropna(subset=['Capacity'])
    discharge_meta = discharge_meta.reset_index(drop=True)

    capacity = discharge_meta['Capacity'].values
    SOH = capacity / capacity[0]

    plt.figure(figsize=(8, 4))
    plt.plot(SOH, linewidth=2)
    plt.xlabel("Discharge Cycle Index")
    plt.ylabel("SOH")
    plt.title("Global Battery SOH Trend")
    plt.grid(True)
    plt.show()

def plot_single_battery_soh(metadata_path, battery_id=None):
    """
    Plots SOH degradation for a single battery.
    If battery_id is None, picks the first battery.
    """
    metadata = pd.read_csv(metadata_path)

    discharge_meta = metadata[
        metadata['type'].str.lower() == 'discharge'
    ].copy()

    discharge_meta['Capacity'] = pd.to_numeric(
        discharge_meta['Capacity'],
        errors='coerce'
    )

    discharge_meta = discharge_meta.dropna(subset=['Capacity'])

    # Auto-pick battery if not provided
    if battery_id is None:
        battery_id = discharge_meta['battery_id'].iloc[0]

    battery_df = discharge_meta[
        discharge_meta['battery_id'] == battery_id
    ].copy()

    battery_df = battery_df.sort_values('start_time').reset_index(drop=True)

    capacity = battery_df['Capacity'].values
    SOH = capacity / capacity[0]

    plt.figure(figsize=(6, 4))
    plt.plot(SOH, marker='o')
    plt.xlabel("Discharge Cycle")
    plt.ylabel("SOH")
    plt.title(f"SOH Degradation - Battery {battery_id}")
    plt.grid(True)
    plt.show()

def compute_coulomb_soh(data_folder, discharge_files):
    capacities = []

    for fname in discharge_files:
        file_path = os.path.join(data_folder, fname)
        df = pd.read_csv(file_path)

        cap = coulomb_capacity(
            df['Time'].values,
            df['Current_measured'].values
        )
        capacities.append(cap)

    capacities = np.array(capacities)
    SOH_cc = capacities / capacities[0]

    plt.figure(figsize=(8, 4))
    plt.plot(SOH_cc, marker='o')
    plt.xlabel("Discharge Cycle Index")
    plt.ylabel("SOH")
    plt.title("SOH via Coulomb Counting")
    plt.grid(True)
    plt.show()

    return SOH_cc

if __name__ == "__main__":
    metadata_path = "cleaned_dataset/metadata.csv"
    data_folder = "cleaned_dataset/data"

    metadata = pd.read_csv(metadata_path)

    discharge_meta = metadata[
        metadata['type'].str.lower() == 'discharge'
    ]

    discharge_files = discharge_meta['filename'].astype(str).tolist()

    # Run separately
    # plot_global_soh(metadata_path)
    # plot_single_battery_soh(metadata_path, battery_id='B0046')

    compute_coulomb_soh(data_folder, discharge_files)
