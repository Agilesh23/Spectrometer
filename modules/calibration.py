from typing import Dict, Optional
import json
import os
import numpy as np

CALIB_PATH = os.path.join("calib", "calibration.json")


def load_calibration_profiles() -> Dict[str, Dict]:
    if not os.path.exists(CALIB_PATH):
        return {}
    try:
        with open(CALIB_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
            if isinstance(data, dict):
                return data
            return {}
    except Exception:
        return {}


def save_calibration_profiles(profiles: Dict[str, Dict]):
    os.makedirs(os.path.dirname(CALIB_PATH), exist_ok=True)
    with open(CALIB_PATH, "w", encoding="utf-8") as f:
        json.dump(profiles, f, indent=2)


def apply_calibration_axis(x: np.ndarray, profile: Optional[Dict]) -> np.ndarray:
    if not profile or "coeffs" not in profile:
        return x
    coeffs = np.array(profile["coeffs"])  # highest degree first for polyval
    return np.polyval(coeffs, x)


def fit_calibration(measured_positions: np.ndarray, known_values: np.ndarray, degree: int = 1):
    degree = int(degree)
    degree = max(1, min(3, degree))
    coeffs = np.polyfit(measured_positions, known_values, degree)
    return coeffs
