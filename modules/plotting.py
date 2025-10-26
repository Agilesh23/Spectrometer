from typing import Optional, List, Dict
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot_spectrum(x: np.ndarray, y: np.ndarray, peaks: Optional[np.ndarray] = None, overlay: Optional[List[Dict]] = None):
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(x, y, label="current", color="#1f77b4")

    if overlay:
        for spec in overlay:
            ax.plot(spec["x"], spec["y"], alpha=0.5, label=spec.get("label", "overlay"))

    if peaks is not None and len(peaks) > 0:
        ax.scatter(peaks, np.interp(peaks, x, y), marker="x", color="red", label="peaks")

    ax.set_xlabel("Axis (pixels or calibrated)")
    ax.set_ylabel("Intensity (a.u.)")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig
