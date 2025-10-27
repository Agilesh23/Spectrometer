from __future__ import annotations
from typing import Dict, Tuple
import numpy as np


def _odd(n: int) -> int:
    return n if n % 2 == 1 else n + 1


def estimate_savgol(y: np.ndarray) -> Tuple[int, int]:
    n = max(5, int(len(y)))
    if n < 11:
        return 5, 2
    # Noise estimate using median absolute deviation of first differences
    dy = np.diff(y)
    mad = np.median(np.abs(dy - np.median(dy))) if dy.size > 0 else 0.0
    noise = 1.4826 * mad
    # Window proportional to length, smaller if high noise
    base = max(11, min(301, _odd(int(n / 50))))
    if noise > 0:
        scale = 1.0 / (1.0 + np.log1p(noise))
    else:
        scale = 1.0
    win = _odd(int(base * scale))
    win = max(5, min(301, win))
    # Polyorder: 2 for short/noisy, 3 otherwise
    poly = 2 if (win <= 21) else 3
    poly = min(poly, win - 1)
    return win, poly


def estimate_background_kernel(y: np.ndarray) -> int:
    n = max(5, int(len(y)))
    k = _odd(int(n / 40))  # broader for longer spectra
    k = max(5, min(301, k))
    return k


def estimate_peak_params(y: np.ndarray) -> Dict[str, float]:
    # Baseline estimate
    baseline = np.median(y)
    resid = y - baseline
    std = float(np.std(resid)) if resid.size else 1.0
    prominence = max(5.0, 3.0 * std)
    distance = max(5, int(len(y) / 60))
    width = max(1, int(len(y) / 400))
    return {
        "prominence": float(prominence),
        "distance": int(distance),
        "width": int(width),
    }


def suggest_all(y: np.ndarray) -> Dict:
    win, poly = estimate_savgol(y)
    bg_kernel = estimate_background_kernel(y)
    peaks = estimate_peak_params(y)
    # Model heuristic placeholder: gaussian default
    model = "gaussian"
    return {
        "win_len": int(win),
        "polyorder": int(poly),
        "bg_sub": True,
        "bg_kernel": int(bg_kernel),
        "prominence": float(peaks["prominence"]),
        "distance": int(peaks["distance"]),
        "width": int(peaks["width"]),
        "do_fit": True,
        "model_choice": model,
    }
