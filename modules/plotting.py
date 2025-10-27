from typing import Optional, List, Dict
import numpy as np
import plotly.graph_objects as go


def plot_spectrum(
    x: np.ndarray,
    y: np.ndarray,
    peaks: Optional[np.ndarray] = None,
    overlay: Optional[List[Dict]] = None,
    peaks_auto: Optional[np.ndarray] = None,
    peaks_manual: Optional[np.ndarray] = None,
):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=y, mode="lines", name="current", line=dict(color="#1f77b4")))

    if overlay:
        for spec in overlay:
            fig.add_trace(
                go.Scatter(
                    x=spec["x"],
                    y=spec["y"],
                    mode="lines",
                    name=str(spec.get("label", "overlay")),
                    opacity=float(spec.get("opacity", 0.5)),
                )
            )

    # Back-compat single 'peaks'
    if peaks is not None and len(peaks) > 0 and (peaks_auto is None and peaks_manual is None):
        fig.add_trace(
            go.Scatter(
                x=peaks,
                y=np.interp(peaks, x, y),
                mode="markers",
                marker=dict(color="red", symbol="x", size=10),
                name="peaks",
            )
        )
    # Separate auto vs manual peaks visualization
    if peaks_auto is not None and len(peaks_auto) > 0:
        fig.add_trace(
            go.Scatter(
                x=peaks_auto,
                y=np.interp(peaks_auto, x, y),
                mode="markers",
                marker=dict(color="#EF553B", symbol="x", size=10),
                name="auto peaks",
            )
        )
    if peaks_manual is not None and len(peaks_manual) > 0:
        fig.add_trace(
            go.Scatter(
                x=peaks_manual,
                y=np.interp(peaks_manual, x, y),
                mode="markers",
                marker=dict(color="#00CC96", symbol="circle-open", size=10, line=dict(width=2)),
                name="manual peaks",
            )
        )

    fig.update_layout(
        margin=dict(l=40, r=20, t=20, b=40),
        template="plotly_white",
        xaxis_title="Axis (pixels or calibrated)",
        yaxis_title="Intensity (a.u.)",
        showlegend=True,
        hovermode="x",
    )
    fig.update_xaxes(showspikes=True, spikemode="across", spikethickness=1)
    fig.update_yaxes(showspikes=True, spikethickness=1)
    return fig
