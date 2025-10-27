import os
import io
import csv
from typing import Optional
import numpy as np
import pandas as pd
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib.units import inch
from reportlab.lib.utils import ImageReader
import plotly

EXPORTS_DIR = "exports"
REPORTS_DIR = "reports"

os.makedirs(EXPORTS_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)


def export_csv(label: str, x: np.ndarray, y: np.ndarray, peaks_df: pd.DataFrame):
    spec_path = os.path.join(EXPORTS_DIR, f"{label}_spectrum.csv")
    peaks_path = os.path.join(EXPORTS_DIR, f"{label}_peaks.csv")

    spec_df = pd.DataFrame({"x": x, "y": y})
    spec_df.to_csv(spec_path, index=False)
    if peaks_df is not None and len(peaks_df) > 0:
        peaks_df.to_csv(peaks_path, index=False)


def export_plot_png(label: str, fig):
    path = os.path.join(EXPORTS_DIR, f"{label}_plot.png")
    if hasattr(fig, "write_image"):
        fig.write_image(path, scale=2)
    elif hasattr(fig, "savefig"):
        fig.savefig(path, dpi=200)
    else:
        raise TypeError("Unsupported figure type for export_plot_png")


def save_processed_image(label: str, img_bgr: np.ndarray):
    import cv2
    path = os.path.join(EXPORTS_DIR, f"{label}_processed.png")
    cv2.imwrite(path, img_bgr)


def export_pdf_report(label: str, fig, x: np.ndarray, y: np.ndarray, peaks_df: pd.DataFrame, calibration: Optional[dict] = None):
    buf = io.BytesIO()
    if hasattr(fig, "to_image"):
        img_bytes = fig.to_image(format="png", scale=2)
        buf.write(img_bytes)
        buf.seek(0)
    elif hasattr(fig, "savefig"):
        fig.savefig(buf, format="png", dpi=200)
        buf.seek(0)
    else:
        raise TypeError("Unsupported figure type for export_pdf_report")

    pdf_path = os.path.join(REPORTS_DIR, f"{label}_report.pdf")
    c = canvas.Canvas(pdf_path, pagesize=A4)
    width, height = A4

    c.setFont("Helvetica-Bold", 16)
    c.drawString(72, height - 72, "Spectrometer Report")
    c.setFont("Helvetica", 10)
    c.drawString(72, height - 90, f"Label: {label}")

    # Plot image
    img = ImageReader(buf)
    img_w = width - 2 * 72
    img_h = img_w * 0.5
    c.drawImage(img, 72, height - 120 - img_h, width=img_w, height=img_h, preserveAspectRatio=True)

    # Peaks table (first N)
    y_cursor = height - 140 - img_h
    c.setFont("Helvetica-Bold", 12)
    c.drawString(72, y_cursor, "Detected Peaks")
    y_cursor -= 16
    c.setFont("Helvetica", 9)

    if peaks_df is not None and len(peaks_df) > 0:
        c.drawString(72, y_cursor, "position")
        c.drawString(200, y_cursor, "amplitude")
        y_cursor -= 12
        max_rows = 20
        for _, row in peaks_df.head(max_rows).iterrows():
            c.drawString(72, y_cursor, f"{row['position']:.3f}")
            c.drawString(200, y_cursor, f"{row['amplitude']:.3f}")
            y_cursor -= 12
    else:
        c.drawString(72, y_cursor, "No peaks detected.")
        y_cursor -= 12

    # Calibration info
    c.setFont("Helvetica-Bold", 12)
    c.drawString(72, y_cursor - 8, "Calibration")
    y_cursor -= 24
    c.setFont("Helvetica", 9)
    if calibration and 'coeffs' in calibration:
        coeffs = calibration['coeffs']
        c.drawString(72, y_cursor, f"Coefficients (high->low): {coeffs}")
        y_cursor -= 12
    else:
        c.drawString(72, y_cursor, "None applied")
        y_cursor -= 12

    c.showPage()
    c.save()
