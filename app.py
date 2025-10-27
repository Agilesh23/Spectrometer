import io
import os
import json
import time
from typing import Optional, List, Dict

import numpy as np
import pandas as pd
import streamlit as st
from streamlit_plotly_events import plotly_events
from streamlit_drawable_canvas import st_canvas
import plotly.graph_objects as go

from modules.image_processing import (
    load_image_from_upload,
    load_image_from_camera,
    apply_rotation,
    apply_crop_margins,
    adjust_brightness_contrast,
    to_grayscale,
    collapse_vertical,
    HAS_CV2,
)
from modules.smoothing import smooth_savgol, subtract_background_median
from modules.peaks import detect_peaks, fit_peaks
from modules.calibration import (
    load_calibration_profiles,
    save_calibration_profiles,
    apply_calibration_axis,
    fit_calibration,
)
from modules.auto_params import suggest_all
from modules.plotting import plot_spectrum
from modules.exports import export_csv, export_plot_png, export_pdf_report, save_processed_image
from modules.utils import ensure_dirs

st.set_page_config(page_title="Pi Spectrometer", layout="wide")

ensure_dirs(["exports", "reports", "calib", "data"]) 

if "spectra" not in st.session_state:
    st.session_state.spectra = []  # list of dicts with keys: x, y, peaks_df, label, opacity
if "calibration" not in st.session_state:
    st.session_state.calibration = {"active": False, "degree": 1, "profile": None, "profiles": {}, "unit": "nm"}
if "manual_peaks" not in st.session_state:
    st.session_state.manual_peaks = {"add": [], "remove": []}
if "overlay" not in st.session_state:
    st.session_state.overlay = False
if "overlay_opacity" not in st.session_state:
    st.session_state.overlay_opacity = 0.5

st.title("Spectrometer")
st.caption("Offline, IT-friendly GUI — no physics jargon")

# Theme and global UI options
cols_top = st.columns([2,1,1])
with cols_top[0]:
    st.markdown("### Instrument Console")
with cols_top[1]:
    theme = st.selectbox("Theme", ["Light", "Dark"], index=0)
with cols_top[2]:
    st.session_state.overlay = st.toggle("Overlay mode", value=st.session_state.get("overlay", False))

plotly_template = "plotly_white" if theme == "Light" else "plotly_dark"

# Section: Image Input & Preprocessing
st.markdown("---")
st.header("Image Input & Preprocessing")

src_options = ["Upload"] if not HAS_CV2 else ["Upload", "Camera"]
src_choice = st.radio("Source", src_options, horizontal=True)
uploaded = None
frame = None
if src_choice == "Upload":
    uploaded = st.file_uploader("Upload image (.jpg, .png)", type=["jpg", "jpeg", "png"]) 
else:
    cam_cols = st.columns([2,1])
    with cam_cols[0]:
        cam_index = st.number_input("Camera index", value=0, step=1)
    with cam_cols[1]:
        if st.button("Capture frame"):
            frame = load_image_from_camera(int(cam_index))
            if frame is None:
                st.error("No camera frame captured.")

pp_cols = st.columns(3)
with pp_cols[0]:
    angle = st.slider("Rotate (deg)", -180, 180, 0)
with pp_cols[1]:
    brightness = st.slider("Brightness", -100, 100, 0)
with pp_cols[2]:
    contrast = st.slider("Contrast", -50, 100, 0)

st.caption("Draw a rectangle on the original image to crop (interactive ROI)")

pre_exp = st.expander("Image Input & Preprocessing", expanded=True)
spec_exp = st.expander("Spectrum Extraction & Peaks", expanded=True)
cal_exp = st.expander("Calibration", expanded=True)
exp_exp = st.expander("Overlay & Export", expanded=True)

# Load image
image = None
if uploaded is not None:
    image = load_image_from_upload(uploaded)
elif frame is not None:
    image = frame

processed_bgr = None
if image is not None:
    try:
        with st.spinner("Applying preprocessing..."):
            rotated = apply_rotation(image, angle)
            if "_roi_rect" in st.session_state:
                lx, ty, ww, hh = st.session_state._roi_rect
                h_img, w_img = rotated.shape[:2]
                x1 = max(0, int(lx))
                y1 = max(0, int(ty))
                x2 = min(w_img, int(lx + ww))
                y2 = min(h_img, int(ty + hh))
                cropped = rotated[y1:y2, x1:x2] if (x2 > x1 and y2 > y1) else rotated
            else:
                cropped = rotated
            adjusted = adjust_brightness_contrast(cropped, brightness, contrast)
            processed_bgr = adjusted
    except ImportError as e:
        st.error(f"Image processing requires OpenCV: {e}")
    except Exception as e:
        st.error(f"Processing failed: {e}")

with pre_exp:
    st.subheader("Image preview")
    if processed_bgr is not None:
        st.image(processed_bgr[:, :, ::-1], channels="BGR", use_column_width=True)
    else:
        st.info("Upload or capture an image to begin.")
    if image is not None:
        st.caption("Optional: draw a rectangle to define ROI for cropping")
        canvas_res = st_canvas(
            fill_color="rgba(0,0,0,0)",
            stroke_width=2,
            stroke_color="#1f77b4",
            background_image=None,
            update_streamlit=True,
            height=min(500, image.shape[0]),
            width=min(800, image.shape[1]),
            drawing_mode="rect",
            key="roi_canvas",
        )
        if canvas_res and canvas_res.json_data is not None and len(canvas_res.json_data.get("objects", [])) > 0:
            obj = canvas_res.json_data["objects"][-1]
            left = int(obj.get("left", 0))
            top = int(obj.get("top", 0))
            width_rect = int(obj.get("width", 0) * obj.get("scaleX", 1))
            height_rect = int(obj.get("height", 0) * obj.get("scaleY", 1))
            st.session_state._roi_rect = (left, top, width_rect, height_rect)

# Spectrum processing pipeline
spectrum_x = None
spectrum_y = None
peaks_df = pd.DataFrame()

if processed_bgr is not None:
    try:
        with st.spinner("Extracting spectrum and detecting peaks..."):
            gray = to_grayscale(processed_bgr)
            x = np.arange(gray.shape[1])
            y = collapse_vertical(gray, mode="sum")
            # Auto mode suggestion
            if st.session_state.auto_mode:
                sug = suggest_all(y.astype(float))
                st.session_state.params.update(sug)
            win_len = int(st.session_state.params["win_len"])
            polyorder = int(st.session_state.params["polyorder"])
            bg_sub = bool(st.session_state.params["bg_sub"]) 
            bg_kernel = int(st.session_state.params["bg_kernel"]) 
            prominence = float(st.session_state.params["prominence"]) 
            distance = int(st.session_state.params["distance"]) 
            width = int(st.session_state.params["width"]) 
            do_fit = bool(st.session_state.params["do_fit"]) 
            model_choice = str(st.session_state.params["model_choice"]) 
            fit_width = int(st.session_state.params["fit_width"]) 

            y_proc = y.astype(float).copy()
            if bg_sub:
                y_proc = subtract_background_median(y_proc, kernel=bg_kernel)
            y_smooth = smooth_savgol(y_proc, window_length=win_len, polyorder=polyorder)
            peak_idx, peak_props = detect_peaks(y_smooth, prominence=prominence, distance=distance, width=width)
            fit_results = None
            if do_fit and len(peak_idx) > 0:
                fit_results = fit_peaks(x, y_smooth, peak_idx, window=fit_width, model=model_choice)
        if st.session_state.calibration["active"]:
            x_disp = apply_calibration_axis(x, st.session_state.calibration.get("profile"))
        else:
            x_disp = x
        if do_fit and fit_results is not None:
            auto_centers = np.array([fr["center"] for fr in fit_results], dtype=float)
            auto_amps = np.array([fr["amplitude"] for fr in fit_results], dtype=float)
        else:
            auto_centers = x_disp[peak_idx].astype(float)
            auto_amps = y_smooth[peak_idx].astype(float)
        added = np.array(st.session_state.manual_peaks.get("add", []), dtype=float) if len(st.session_state.manual_peaks.get("add", []))>0 else np.array([], dtype=float)
        removed = np.array(st.session_state.manual_peaks.get("remove", []), dtype=float) if len(st.session_state.manual_peaks.get("remove", []))>0 else np.array([], dtype=float)
        tol = max(1.0, (x_disp.max() - x_disp.min()) / 500.0)
        # Remove auto peaks near any removed
        if auto_centers.size > 0 and removed.size > 0:
            keep = np.ones_like(auto_centers, dtype=bool)
            for rv in removed:
                keep &= (np.abs(auto_centers - rv) > tol)
            auto_centers = auto_centers[keep]
            auto_amps = auto_amps[keep]
        manual_centers = np.array(sorted(list(added))) if added.size > 0 else np.array([], dtype=float)
        manual_amps = np.interp(manual_centers, x_disp, y_smooth) if manual_centers.size > 0 else np.array([], dtype=float)
        # Combined peaks table with source labels
        pos_list = []
        amp_list = []
        src_list = []
        if auto_centers.size > 0:
            pos_list.extend(list(auto_centers))
            amp_list.extend(list(auto_amps))
            src_list.extend(["auto"] * len(auto_centers))
        if manual_centers.size > 0:
            pos_list.extend(list(manual_centers))
            amp_list.extend(list(manual_amps))
            src_list.extend(["manual"] * len(manual_centers))
        if len(pos_list) > 0:
            peaks_df = pd.DataFrame({"position": np.array(pos_list), "amplitude": np.array(amp_list), "source": np.array(src_list)})
        else:
            peaks_df = pd.DataFrame(columns=["position", "amplitude", "source"])            
        spectrum_x, spectrum_y = x_disp, y_smooth
    except ImportError as e:
        st.error(f"Spectrum extraction requires OpenCV: {e}")
    except Exception as e:
        st.error(f"Spectrum processing failed: {e}")

with spec_exp:
    if "params" not in st.session_state:
        st.session_state.params = {
            "win_len": 31,
            "polyorder": 3,
            "bg_sub": False,
            "bg_kernel": 51,
            "prominence": 20.0,
            "distance": 10,
            "width": 1,
            "fit_width": 40,
            "do_fit": False,
            "model_choice": "gaussian",
        }
    if "auto_mode" not in st.session_state:
        st.session_state.auto_mode = False

    top_cols = st.columns([1,1,1,1,1,1])
    with top_cols[0]:
        st.session_state.auto_mode = st.checkbox("Auto mode", value=st.session_state.auto_mode, help="Auto-select smoothing, background, and peak params")
    with top_cols[1]:
        if st.button("Suggest parameters") and processed_bgr is not None:
            try:
                gray_tmp = to_grayscale(processed_bgr)
                y_tmp = collapse_vertical(gray_tmp, mode="sum").astype(float)
                sug = suggest_all(y_tmp)
                st.session_state.params.update(sug)
                st.success("Suggested parameters applied")
                st.experimental_rerun()
            except Exception as e:
                st.error(f"Auto suggestion failed: {e}")

    ctrl_cols = st.columns(6)
    with ctrl_cols[0]:
        st.session_state.params["win_len"] = st.slider("Savitzky–Golay window", 5, 301, int(st.session_state.params["win_len"]), step=2, help="Smoothing window (odd)")
    with ctrl_cols[1]:
        st.session_state.params["polyorder"] = st.slider("Savgol polyorder", 1, 5, int(st.session_state.params["polyorder"]))
    with ctrl_cols[2]:
        st.session_state.params["bg_sub"] = st.checkbox("Background subtraction", value=bool(st.session_state.params["bg_sub"]))
    with ctrl_cols[3]:
        st.session_state.params["bg_kernel"] = st.slider("BG kernel (odd)", 5, 301, int(st.session_state.params["bg_kernel"]), step=2)
    with ctrl_cols[4]:
        st.session_state.params["model_choice"] = st.selectbox("Fit model", ["gaussian", "lorentzian"], index=0 if st.session_state.params["model_choice"]=="gaussian" else 1)
    with ctrl_cols[5]:
        st.session_state.params["fit_width"] = st.slider("Fit window (px)", 5, 200, int(st.session_state.params["fit_width"]))
    pk_cols = st.columns(4)
    with pk_cols[0]:
        st.session_state.params["prominence"] = st.slider("Prominence", 0.0, 1000.0, float(st.session_state.params["prominence"]))
    with pk_cols[1]:
        st.session_state.params["distance"] = st.slider("Min distance (px)", 1, 200, int(st.session_state.params["distance"]))
    with pk_cols[2]:
        st.session_state.params["width"] = st.slider("Min width (px)", 1, 100, int(st.session_state.params["width"]))
    with pk_cols[3]:
        st.session_state.params["do_fit"] = st.checkbox("Peak fitting", value=bool(st.session_state.params["do_fit"]))
    st.subheader("Spectrum")
    if spectrum_x is not None:
        peaks_auto = auto_centers if 'auto_centers' in locals() else None
        peaks_manual = manual_centers if 'manual_centers' in locals() else None
        fig = plot_spectrum(
            spectrum_x,
            spectrum_y,
            peaks=None,
            overlay=st.session_state.spectra if st.session_state.get("overlay", False) else None,
            peaks_auto=peaks_auto,
            peaks_manual=peaks_manual,
        )
        fig.update_layout(template=plotly_template)
        edit = st.toggle("Edit peaks manually", value=False, help="Click on the plot to add/remove peaks")
        if edit:
            mode = st.radio("Mode", ["Add", "Remove"], horizontal=True)
            clicked = plotly_events(fig, click_event=True, hover_event=False, select_event=False, key="pe_click")
            st.plotly_chart(fig, use_container_width=True)
            if clicked:
                x_click = float(clicked[0]["x"])  # type: ignore
                if mode == "Add":
                    st.session_state.manual_peaks["add"].append(x_click)
                else:
                    st.session_state.manual_peaks["remove"].append(x_click)
            c1, c2 = st.columns(2)
            with c1:
                if st.button("Clear added"):
                    st.session_state.manual_peaks["add"] = []
            with c2:
                if st.button("Clear removed"):
                    st.session_state.manual_peaks["remove"] = []
        else:
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Processed spectrum will appear here.")

# Peaks table and manual calibration mapping
st.subheader("Detected peaks")
st.dataframe(peaks_df, use_container_width=True, hide_index=True)

with cal_exp:
    st.subheader("Calibration")
    profiles = load_calibration_profiles()
    st.session_state.calibration["profiles"] = profiles
    cal_top = st.columns(4)
    with cal_top[0]:
        cal_names = ["(none)"] + list(profiles.keys())
        sel = st.selectbox("Profile", cal_names)
    with cal_top[1]:
        active = st.toggle("Apply calibration", value=st.session_state.calibration.get("active", False))
    with cal_top[2]:
        degree = st.selectbox("Degree", [1,2,3], index=[1,2,3].index(st.session_state.calibration.get("degree",1)))
    with cal_top[3]:
        st.session_state.calibration["unit"] = st.selectbox("Units", ["nm","cm^-1","custom"], index=["nm","cm^-1","custom"].index(st.session_state.calibration.get("unit","nm")))
    st.session_state.calibration["active"] = active
    st.session_state.calibration["degree"] = degree
    st.session_state.calibration["profile"] = None if sel == "(none)" else profiles.get(sel)
    if "cal_points" not in st.session_state:
        st.session_state.cal_points = []  # list of dicts {x, known}
    if spectrum_x is not None and len(peaks_df) > 0:
        st.caption("Click on peaks to add points. Assign known wavelengths, then fit and save.")
        cal_fig = plot_spectrum(
            spectrum_x,
            spectrum_y,
            peaks=peaks_df["position"].values if len(peaks_df)>0 else None,
            overlay=None,
        )
        cal_fig.update_layout(template=plotly_template)
        clicks = plotly_events(cal_fig, click_event=True, hover_event=False, select_event=False, key="cal_click")
        st.plotly_chart(cal_fig, use_container_width=True)
        if clicks:
            st.session_state.cal_points.append({"x": float(clicks[0]["x"]), "known": None})
        if len(st.session_state.cal_points) > 0:
            df_points = pd.DataFrame(st.session_state.cal_points)
            st.write("Selected points:")
            for i in range(len(df_points)):
                c1, c2, c3 = st.columns([2,2,1])
                with c1:
                    st.text_input("Measured x", value=f"{df_points.iloc[i]['x']:.4f}", key=f"cal_x_{i}", disabled=True)
                with c2:
                    val = st.text_input("Known", value="" if pd.isna(df_points.iloc[i]['known']) or df_points.iloc[i]['known'] is None else str(df_points.iloc[i]['known']), key=f"cal_known_{i}")
                    try:
                        st.session_state.cal_points[i]["known"] = float(val) if val.strip() != "" else None
                    except Exception:
                        st.session_state.cal_points[i]["known"] = None
                with c3:
                    if st.button("Remove", key=f"cal_rm_{i}"):
                        st.session_state.cal_points.pop(i)
                        st.experimental_rerun()
            degree = st.session_state.calibration["degree"]
            prof_name = st.text_input("Profile name", value="my_light_source")
            unit = st.selectbox("Units", ["nm", "cm^-1", "custom"], index=["nm","cm^-1","custom"].index(st.session_state.calibration.get("unit","nm")))
            cfit1, cfit2 = st.columns(2)
            with cfit1:
                if st.button("Preview fit"):
                    pass
            xs = np.array([p["x"] for p in st.session_state.cal_points if p["known"] is not None], dtype=float)
            ys = np.array([p["known"] for p in st.session_state.cal_points if p["known"] is not None], dtype=float)
            if len(xs) >= 2 and len(xs) == len(ys):
                coeffs = fit_calibration(xs, ys, degree=degree)
                y_pred = np.polyval(coeffs, xs)
                rmse = float(np.sqrt(np.mean((ys - y_pred) ** 2)))
                st.session_state._cal_preview = {"coeffs": coeffs, "rmse": rmse}
                st.write(f"RMSE: {rmse:.4f} {unit}")
                resid_fig = go.Figure()
                resid_fig.add_trace(go.Scatter(x=xs, y=ys - y_pred, mode="markers", name="residuals"))
                resid_fig.update_layout(template="plotly_white", xaxis_title="Measured x", yaxis_title="Residual (known - fit)")
                st.plotly_chart(resid_fig, use_container_width=True)
                if st.button("Save profile"):
                    profiles = st.session_state.calibration["profiles"]
                    profiles[prof_name] = {"coeffs": coeffs.tolist(), "unit": unit, "rmse": rmse}
                    save_calibration_profiles(profiles)
                    st.success(f"Saved calibration profile '{prof_name}'. Reload from sidebar.")
            else:
                st.info("Add at least 2 valid points with known values to fit.")
        else:
            st.info("Click on the plot to add calibration points.")
    else:
        st.info("Run detection to populate peaks first.")

# Overlay and export controls
with exp_exp:
    export_label = st.text_input("Spectrum label", value=f"run-{int(time.time())}")
    st.session_state.overlay_opacity = st.slider("Default overlay opacity", 0.1, 1.0, st.session_state.overlay_opacity)
    # Preview of current export plot
    if spectrum_x is not None:
        preview_fig = plot_spectrum(
            spectrum_x,
            spectrum_y,
            peaks=None,
            overlay=st.session_state.spectra if st.session_state.get("overlay", False) else None,
            peaks_auto=(auto_centers if 'auto_centers' in locals() else None),
            peaks_manual=(manual_centers if 'manual_centers' in locals() else None),
        )
        preview_fig.update_layout(template=plotly_template)
        st.plotly_chart(preview_fig, use_container_width=True)
    colA, colB, colC, colD = st.columns(4)
    with colA:
        if spectrum_x is not None and st.button("Add to overlay"):
            st.session_state.spectra.append({
                "x": spectrum_x.copy(),
                "y": spectrum_y.copy(),
                "peaks_df": peaks_df.copy(),
                "label": export_label,
                "opacity": st.session_state.overlay_opacity,
            })
            st.success("Spectrum added to overlay list.")
    with colB:
        if spectrum_x is not None and st.button("Export CSV"):
            export_csv(export_label, spectrum_x, spectrum_y, peaks_df)
            st.success("Saved CSV to exports/.")
    with colC:
        if spectrum_x is not None and st.button("Export PNG plot"):
            fig = plot_spectrum(spectrum_x, spectrum_y, peaks=peaks_df["position"].values if len(peaks_df)>0 else None, overlay=st.session_state.spectra if st.session_state.get("overlay", False) else None)
            fig.update_layout(template=plotly_template)
            export_plot_png(export_label, fig)
            st.success("Saved plot PNG to exports/.")
    with colD:
        if processed_bgr is not None and st.button("Save processed image"):
            try:
                save_processed_image(export_label, processed_bgr)
                st.success("Saved processed image to exports/.")
            except ImportError as e:
                st.error(f"Saving image requires OpenCV: {e}")
            except Exception as e:
                st.error(f"Failed to save image: {e}")
    if spectrum_x is not None and st.button("Export PDF report"):
        fig = plot_spectrum(spectrum_x, spectrum_y, peaks=peaks_df["position"].values if len(peaks_df)>0 else None, overlay=st.session_state.spectra if st.session_state.get("overlay", False) else None)
        fig.update_layout(template=plotly_template)
        active_prof = st.session_state.calibration.get("profile")
        export_pdf_report(export_label, fig, spectrum_x, spectrum_y, peaks_df, calibration=active_prof)
        st.success("Saved PDF to reports/.")
    if len(st.session_state.spectra) > 0:
        st.subheader("Overlay items")
        for i, spec in enumerate(st.session_state.spectra):
            c1, c2, c3 = st.columns([3,1,1])
            with c1:
                st.text_input("Label", value=str(spec.get("label","overlay")), key=f"ov_lbl_{i}", disabled=True)
            with c2:
                spec["opacity"] = st.slider("Opacity", 0.1, 1.0, float(spec.get("opacity", 0.5)), key=f"ov_op_{i}")
            with c3:
                if st.button("Remove", key=f"ov_rm_{i}"):
                    st.session_state.spectra.pop(i)
                    st.experimental_rerun()

# Status bar
st.markdown("---")
status_cols = st.columns(4)
with status_cols[0]:
    st.write(f"Image: {'loaded' if image is not None else 'none'}")
with status_cols[1]:
    st.write(f"Peaks: {len(peaks_df) if isinstance(peaks_df, pd.DataFrame) else 0}")
with status_cols[2]:
    st.write(f"Calibration: {'on' if st.session_state.calibration.get('active') else 'off'}")
with status_cols[3]:
    st.write(f"Overlays: {len(st.session_state.spectra)}")
