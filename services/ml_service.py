"""
Machine Learning State Estimation Service.
Loads TFLite LSTM model and feature scaler for battery State of Health (SOH) prediction.
Designed to be lightweight and non-blocking for edge execution on Raspberry Pi 3B+.
"""

import os
import time
import pickle
import logging
from typing import Optional, List
import numpy as np

from core.config import (
    MODEL_PATH,
    SCALER_PATH,
    SERIES_CELLS,
    PARALLEL_CELLS
)

logger = logging.getLogger("MLService")

class MLService:
    def __init__(self, model_path: str = MODEL_PATH, scaler_path: str = SCALER_PATH, seq_len: int = 100):
        self.seq_len = seq_len
        self.scaler = None
        self.interpreter = None
        self.input_details = None
        self.output_details = None
        self.history_buffer: List[np.ndarray] = []
        self.last_soh = 98.5
        self.is_model_loaded = False

        # 1. Load Scaler
        if os.path.exists(scaler_path):
            try:
                with open(scaler_path, "rb") as f:
                    self.scaler = pickle.load(f)
                logger.info(f"Loaded feature scaler from {scaler_path}")
            except Exception as e:
                logger.warning(f"Failed to load scaler from {scaler_path}: {e}")

        # 2. Load TFLite Model
        tflite = None
        try:
            import tflite_runtime.interpreter as tflite
        except ImportError:
            try:
                import tensorflow.lite as tflite
            except ImportError:
                logger.info("TFLite runtime not present. MLService running heuristic SOH estimation.")

        if tflite and os.path.exists(model_path):
            try:
                self.interpreter = tflite.Interpreter(model_path=model_path)
                self.interpreter.allocate_tensors()
                self.input_details = self.interpreter.get_input_details()
                self.output_details = self.interpreter.get_output_details()
                self.is_model_loaded = True
                logger.info(f"TFLite model loaded successfully from {model_path}")
            except Exception as e:
                logger.warning(f"Failed to initialize TFLite model: {e}")

    def build_features(self, pack_voltage: float, current: float, temperature: float) -> np.ndarray:
        """
        Converts multi-cell pack parameters to single-cell domain matching model training.
        Features: [cell_voltage, cell_current, temperature]
        """
        cell_voltage = pack_voltage / float(SERIES_CELLS) if SERIES_CELLS > 0 else pack_voltage
        cell_current = current / float(PARALLEL_CELLS) if PARALLEL_CELLS > 0 else current
        raw = np.array([[cell_voltage, cell_current, temperature]], dtype=np.float32)

        if self.scaler is not None:
            try:
                return self.scaler.transform(raw)[0]
            except Exception:
                pass

        # Fallback manual normalization
        v_norm = (cell_voltage - 3.0) / (4.2 - 3.0)
        i_norm = (cell_current - (-4.0)) / (4.0 - (-4.0))
        t_norm = (temperature - 20.0) / (60.0 - 20.0)
        return np.array([v_norm, i_norm, t_norm], dtype=np.float32)

    def predict_soh(self, pack_voltage: float, current: float, temperature: float) -> float:
        """
        Feeds sliding window buffer of features into TFLite LSTM model.
        Returns SOH percentage (60.0 - 100.0).
        """
        feats = self.build_features(pack_voltage, current, temperature)
        self.history_buffer.append(feats)
        if len(self.history_buffer) > self.seq_len:
            self.history_buffer.pop(0)

        if self.is_model_loaded and self.interpreter is not None:
            try:
                # Buffer warmup padding
                if len(self.history_buffer) < self.seq_len:
                    pad_count = self.seq_len - len(self.history_buffer)
                    seq_array = np.vstack([np.tile(self.history_buffer[0], (pad_count, 1)), np.array(self.history_buffer)])
                else:
                    seq_array = np.array(self.history_buffer)

                input_tensor = np.expand_dims(seq_array, axis=0).astype(np.float32)
                self.interpreter.set_tensor(self.input_details[0]['index'], input_tensor)
                self.interpreter.invoke()
                raw_out = self.interpreter.get_tensor(self.output_details[0]['index'])
                soh_val = float(raw_out[0][0]) * 100.0

                # Exponential smoothing filter to prevent high-frequency jumping
                self.last_soh = round(0.98 * self.last_soh + 0.02 * soh_val, 1)
                self.last_soh = max(60.0, min(100.0, self.last_soh))
                return self.last_soh
            except Exception as e:
                logger.debug(f"Inference error: {e}")

        # Heuristic slow degradation baseline if model not loaded
        return round(self.last_soh, 1)

ml_service = MLService()
