import numpy as np
from scipy.signal import savgol_filter, medfilt


def smooth_savgol(y: np.ndarray, window_length: int = 31, polyorder: int = 3) -> np.ndarray:
    if window_length % 2 == 0:
        window_length += 1
    window_length = max(polyorder + 2, window_length)
    window_length = min(len(y) - 1 if len(y) % 2 == 0 else len(y), window_length)
    if window_length < polyorder + 2:
        return y
    return savgol_filter(y, window_length=window_length, polyorder=polyorder)


def subtract_background_median(y: np.ndarray, kernel: int = 51) -> np.ndarray:
    if kernel % 2 == 0:
        kernel += 1
    baseline = medfilt(y, kernel_size=kernel)
    out = y - baseline
    out[out < 0] = 0
    return out
