from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from vehicle_analyzer.processing import CalibrationConfig, VehicleAnalyzer


st.set_page_config(page_title="Vehicle Speed & Tracking Analyzer", layout="wide")
st.title("🚘 Vehicle Speed & Tracking Analyzer")
st.caption(
    "Upload a video to detect/track vehicles, estimate speed dynamically, and log vehicle type + speed range + optional number plate text."
)

with st.sidebar:
    st.header("Settings")
    confidence = st.slider("Detection confidence", 0.1, 0.9, 0.25, 0.05)
    meters_per_pixel = st.number_input(
        "Fallback meters-per-pixel", min_value=0.001, max_value=10.0, value=0.05, step=0.005
    )
    enable_plate_ocr = st.checkbox("Enable number plate OCR (slower)", value=False)

    st.subheader("Optional perspective calibration")
    st.markdown(
        "Provide JSON arrays with 4 points each. Example image points: `[[100,200],[500,200],[550,400],[50,400]]`"
    )
    image_points_text = st.text_area("Image points (pixels)", value="")
    world_points_text = st.text_area("World points (meters)", value="")

st.markdown(
    "**Tip:** For more accurate speeds, calibrate with 4 points on the road and real-world meter coordinates."
)

uploaded = st.file_uploader("Upload video", type=["mp4", "avi", "mov", "mkv", "webm"])


def parse_points(raw: str):
    if not raw.strip():
        return None
    data = json.loads(raw)
    arr = np.asarray(data, dtype=np.float32)
    if arr.shape != (4, 2):
        raise ValueError("Expected exactly 4 points with shape (4,2)")
    return arr


if uploaded is not None:
    workdir = Path("runs")
    workdir.mkdir(exist_ok=True)

    input_path = workdir / uploaded.name
    output_path = workdir / f"processed_{uploaded.stem}.mp4"
    log_path = workdir / f"log_{uploaded.stem}.csv"

    with open(input_path, "wb") as f:
        f.write(uploaded.read())

    try:
        calibration = CalibrationConfig(
            meters_per_pixel=meters_per_pixel,
            image_points=parse_points(image_points_text),
            world_points=parse_points(world_points_text),
        )
    except Exception as e:
        st.error(f"Invalid calibration points: {e}")
        st.stop()

    if st.button("Analyze video", type="primary"):
        with st.spinner("Processing video... this can take time depending on resolution and duration."):
            analyzer = VehicleAnalyzer(
                confidence=confidence,
                calibration=calibration,
                enable_plate_ocr=enable_plate_ocr,
            )
            df: pd.DataFrame = analyzer.process_video(input_path, output_path)
            df.to_csv(log_path, index=False)

        st.success("Analysis complete")
        st.video(str(output_path))

        st.subheader("Vehicle event log")
        st.dataframe(df, use_container_width=True)

        if not df.empty:
            st.subheader("Summary")
            c1, c2, c3 = st.columns(3)
            c1.metric("Tracked events", len(df))
            c2.metric("Unique vehicles", int(df["track_id"].nunique()))
            valid_speed = df["speed_kmh"].dropna()
            avg_speed = f"{valid_speed.mean():.1f} km/h" if not valid_speed.empty else "N/A"
            c3.metric("Average speed", avg_speed)

        st.download_button(
            "Download log CSV",
            data=log_path.read_bytes(),
            file_name=log_path.name,
            mime="text/csv",
        )

        st.markdown("### Improvements Included")
        st.markdown(
            "- Perspective-aware speed estimation via optional homography\n"
            "- Outlier-robust rolling speed smoothing\n"
            "- Vehicle speed banding (slow/moderate/fast/very fast)\n"
            "- Optional OCR pipeline for visible number plates"
        )
else:
    st.info("Upload a video to begin.")
