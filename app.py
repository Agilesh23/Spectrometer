import io
import os
import json
import time
from typing import Optional, List, Dict

import numpy as np
import pandas as pd
import streamlit as st

from modules.image_processing import (
    load_image_from_upload,
    load_image_from_camera,
    apply_rotation,
    apply_crop_margins,
    adjust_brightness_contrast,
    to_grayscale,
    collapse_vertical,
)
from modules.smoothing import smooth_savgol, subtract_background_median
from modules.peaks import detect_peaks, fit_peaks
from modules.calibration import (
    load_calibration_profiles,
    save_calibration_profiles,
    apply_calibration_axis,
    fit_calibration,
)
from modules.plotting import plot_spectrum
from modules.exports import export_csv, export_plot_png, export_pdf_report, save_processed_image
from modules.utils import ensure_dirs

st.set_page_config(page_title="Pi Spectrometer", layout="wide")

ensure_dirs(["exports", "reports", "calib", "data"]) 

if "spectra" not in st.session_state:
    st.session_state.spectra = []  # list of dicts with keys: x, y, peaks_df, label
if "calibration" not in st.session_state:
    st.session_state.calibration = {"active": False, "degree": 1, "profile": None, "profiles": {}}

st.title("Spectrometer")
st.caption("Offline, IT-friendly GUI — no physics jargon")

with st.sidebar:
    st.header("Image Input")
    src_choice = st.radio("Source", ["Upload", "Camera"], horizontal=True)
    uploaded = None
    frame = None
    if src_choice == "Upload":
        uploaded = st.file_uploader("Upload image (.jpg, .png)", type=["jpg", "jpeg", "png"]) 
    else:
        cam_index = st.number_input("Camera index", value=0, step=1)
        grab = st.button("Capture frame")
        if grab:
            frame = load_image_from_camera(int(cam_index))
            if frame is None:
                st.error("No camera frame captured.")

    st.divider()
    st.header("Preprocess")
    angle = st.slider("Rotate (deg)", -180, 180, 0)
    crop_left = st.slider("Crop left (px)", 0, 2000, 0)
    crop_right = st.slider("Crop right (px)", 0, 2000, 0)
    crop_top = st.slider("Crop top (px)", 0, 2000, 0)
    crop_bottom = st.slider("Crop bottom (px)", 0, 2000, 0)
    brightness = st.slider("Brightness", -100, 100, 0)
    contrast = st.slider("Contrast", -50, 100, 0)

    st.divider()
    st.header("Processing")
    win_len = st.slider("Savitzky–Golay window", 5, 301, 31, step=2)
    polyorder = st.slider("Savgol polyorder", 1, 5, 3)
    bg_sub = st.checkbox("Background subtraction (median)")
    bg_kernel = st.slider("Background kernel (odd)", 5, 301, 51, step=2)

    st.divider()
    st.header("Peaks")
    prominence = st.slider("Prominence", 0.0, 1000.0, 20.0)
    distance = st.slider("Min distance (px)", 1, 200, 10)
    width = st.slider("Min width (px)", 1, 100, 1)
    do_fit = st.checkbox("Peak fitting (Gaussian)")
    fit_width = st.slider("Fit window (px)", 5, 200, 40, step=1)
    model_choice = st.selectbox("Fit model", ["gaussian", "lorentzian"], index=0)

    st.divider()
    st.header("Calibration")
    profiles = load_calibration_profiles()
    st.session_state.calibration["profiles"] = profiles
    cal_names = ["(none)"] + list(profiles.keys())
    sel = st.selectbox("Profile", cal_names)
    active = st.checkbox("Apply calibration", value=False)
    st.session_state.calibration["active"] = active
    st.session_state.calibration["profile"] = None if sel == "(none)" else profiles.get(sel)
    degree = st.selectbox("Calib degree", [1, 2, 3], index=0)
    st.session_state.calibration["degree"] = degree

    st.caption("Manual calibration: map detected peaks to known values below")

    st.divider()
    st.header("Export")
    do_overlay = st.checkbox("Overlay spectra")
    export_label = st.text_input("Spectrum label", value=f"run-{int(time.time())}")

col1, col2 = st.columns([1,1])

# Load image
image = None
if uploaded is not None:
    image = load_image_from_upload(uploaded)
elif frame is not None:
    image = frame

# Process preview
processed_bgr = None
if image is not None:
    rotated = apply_rotation(image, angle)
    cropped = apply_crop_margins(rotated, crop_left, crop_right, crop_top, crop_bottom)
    adjusted = adjust_brightness_contrast(cropped, brightness, contrast)
    processed_bgr = adjusted

with col1:
    st.subheader("Image preview")
    if processed_bgr is not None:
        st.image(processed_bgr[:, :, ::-1], channels="BGR", use_column_width=True)
    else:
        st.info("Upload or capture an image to begin.")

# Spectrum processing pipeline
spectrum_x = None
spectrum_y = None
peaks_df = pd.DataFrame()

if processed_bgr is not None:
    gray = to_grayscale(processed_bgr)
    x = np.arange(gray.shape[1])
    y = collapse_vertical(gray, mode="sum")
    y_proc = y.astype(float).copy()
    if bg_sub:
        y_proc = subtract_background_median(y_proc, kernel=bg_kernel)
    y_smooth = smooth_savgol(y_proc, window_length=win_len, polyorder=polyorder)

    # Detect peaks
    peak_idx, peak_props = detect_peaks(y_smooth, prominence=prominence, distance=distance, width=width)

    # Optional fitting
    fit_results = None
    if do_fit and len(peak_idx) > 0:
        fit_results = fit_peaks(x, y_smooth, peak_idx, window=fit_width, model=model_choice)

    # Calibration
    if st.session_state.calibration["active"]:
        x_disp = apply_calibration_axis(x, st.session_state.calibration.get("profile"))
    else:
        x_disp = x

    # Build peaks table
    if do_fit and fit_results is not None:
        centers = [fr["center"] for fr in fit_results]
        amps = [fr["amplitude"] for fr in fit_results]
    else:
        centers = x_disp[peak_idx]
        amps = y_smooth[peak_idx]
    peaks_df = pd.DataFrame({
        "position": centers,
        "amplitude": amps,
    })

    spectrum_x, spectrum_y = x_disp, y_smooth

with col2:
    st.subheader("Spectrum")
    if spectrum_x is not None:
        fig = plot_spectrum(spectrum_x, spectrum_y, peaks=peaks_df["position"].values if len(peaks_df)>0 else None, overlay=st.session_state.spectra if st.session_state.get("overlay", False) else None)
        st.pyplot(fig, use_container_width=True)
    else:
        st.info("Processed spectrum will appear here.")

# Peaks table and manual calibration mapping
st.subheader("Detected peaks")
st.dataframe(peaks_df, use_container_width=True, hide_index=True)

with st.expander("Manual calibration (map peaks to known values)"):
    if spectrum_x is not None and len(peaks_df) > 0:
        st.caption("Select rows and enter known values, then fit and save as a profile.")
        sel_rows = st.multiselect("Select peak rows", options=list(range(len(peaks_df))), default=list(range(min(3, len(peaks_df)))))
        known_values = st.text_input("Known values for selected peaks (comma-separated)", value="")
        prof_name = st.text_input("Profile name", value="my_light_source")
        if st.button("Fit calibration and save profile"):
            try:
                selected_positions = peaks_df.iloc[sel_rows]["position"].values
                known = np.array([float(v.strip()) for v in known_values.split(",") if v.strip()])
                if len(selected_positions) != len(known) or len(known) < 2:
                    st.error("Provide the same number of known values as selected peaks (>=2).")
                else:
                    coeffs = fit_calibration(selected_positions, known, degree=st.session_state.calibration["degree"]) 
                    profiles = st.session_state.calibration["profiles"]
                    profiles[prof_name] = {"coeffs": coeffs.tolist()}
                    save_calibration_profiles(profiles)
                    st.success(f"Saved calibration profile '{prof_name}'. Reload from sidebar.")
            except Exception as e:
                st.error(f"Calibration failed: {e}")
    else:
        st.info("Run detection to populate peaks first.")

# Overlay and export controls
st.session_state.overlay = do_overlay

colA, colB, colC, colD = st.columns(4)
with colA:
    if spectrum_x is not None and st.button("Add to overlay"):
        st.session_state.spectra.append({
            "x": spectrum_x.copy(),
            "y": spectrum_y.copy(),
            "peaks_df": peaks_df.copy(),
            "label": export_label,
        })
        st.success("Spectrum added to overlay list.")
with colB:
    if spectrum_x is not None and st.button("Export CSV"):
        export_csv(export_label, spectrum_x, spectrum_y, peaks_df)
        st.success("Saved CSV to exports/.")
with colC:
    if spectrum_x is not None and st.button("Export PNG plot"):
        fig = plot_spectrum(spectrum_x, spectrum_y, peaks=peaks_df["position"].values if len(peaks_df)>0 else None, overlay=st.session_state.spectra if st.session_state.get("overlay", False) else None)
        export_plot_png(export_label, fig)
        st.success("Saved plot PNG to exports/.")
with colD:
    if processed_bgr is not None and st.button("Save processed image"):
        save_processed_image(export_label, processed_bgr)
        st.success("Saved processed image to exports/.")

if spectrum_x is not None and st.button("Export PDF report"):
    fig = plot_spectrum(spectrum_x, spectrum_y, peaks=peaks_df["position"].values if len(peaks_df)>0 else None, overlay=st.session_state.spectra if st.session_state.get("overlay", False) else None)
    active_prof = st.session_state.calibration.get("profile")
    export_pdf_report(export_label, fig, spectrum_x, spectrum_y, peaks_df, calibration=active_prof)
    st.success("Saved PDF to reports/.")
