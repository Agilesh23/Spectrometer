from typing import Tuple, List, Dict
import numpy as np
from scipy.signal import find_peaks
from scipy.optimize import curve_fit


def detect_peaks(y: np.ndarray, prominence: float = 10.0, distance: int = 10, width: int = 1) -> Tuple[np.ndarray, Dict]:
    peaks, props = find_peaks(y, prominence=prominence, distance=distance, width=width)
    return peaks, props


def _gaussian(x, amp, mu, sigma, c):
    return amp * np.exp(-0.5 * ((x - mu) / sigma) ** 2) + c


def _lorentzian(x, amp, x0, gamma, c):
    return amp * (gamma ** 2 / ((x - x0) ** 2 + gamma ** 2)) + c


def fit_peaks(x: np.ndarray, y: np.ndarray, peaks_idx: np.ndarray, window: int = 40, model: str = "gaussian") -> List[Dict]:
    results = []
    for p in peaks_idx:
        lo = max(0, p - window)
        hi = min(len(x), p + window + 1)
        xs = x[lo:hi]
        ys = y[lo:hi]
        try:
            if model == "lorentzian":
                p0 = [ys.max() - ys.min(), x[p], max(1.0, window / 5), ys.min()]
                popt, _ = curve_fit(_lorentzian, xs, ys, p0=p0, maxfev=10000)
                amp, center, gamma, c = popt
                results.append({"center": center, "amplitude": amp, "width": gamma, "model": "lorentzian"})
            else:
                p0 = [ys.max() - ys.min(), x[p], max(1.0, window / 5), ys.min()]
                popt, _ = curve_fit(_gaussian, xs, ys, p0=p0, maxfev=10000)
                amp, center, sigma, c = popt
                results.append({"center": center, "amplitude": amp, "width": sigma, "model": "gaussian"})
        except Exception:
            # Fallback to raw peak position
            results.append({"center": x[p], "amplitude": y[p], "width": None, "model": "na"})
    return results
