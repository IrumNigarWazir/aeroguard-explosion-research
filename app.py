"""
AeroGuard AI - UAV Early Warning C2 Dashboard
Author: Irum Nigar Wazir
FYP-II: Real-Time Aerial Hazard Detection
"""

import tempfile
import cv2
import numpy as np
from PIL import Image
import streamlit as st
import time
from ultralytics import YOLO

# 1. Page Configuration (Military/Corporate C2 Styling)
st.set_page_config(
    page_title="AeroGuard AI | UAV Command", page_icon="🚁", layout="wide"
)

st.title("🚁 AeroGuard AI: Real-Time UAV Hazard Dispatch")
st.markdown(
    "**Active Payload:** YOLO11n (FP16) | **Target Sector:** Wide-Area (40+ Kanal) | **Status:** `MONITORING`"
)
st.divider()

# 2. Sidebar Controls
st.sidebar.header("⚙️ Flight Controller Settings")
model_path = st.sidebar.text_input("Checkpoint Path", "best.pt")
conf_thresh = st.sidebar.slider("Confidence Sensitivity", 0.10, 1.00, 0.45, 0.05)
feed_source = st.sidebar.radio(
    "Optical Input Source", ["Live Drone Telemetry (Webcam)", "Pre-Recorded UAV Patrol (MP4)"]
)

run_patrol = st.sidebar.checkbox("▶ INITIATE ACTIVE SCAN", value=False)


# 3. Load YOLO11n Model safely
@st.cache_resource
def load_payload(weights):
    return YOLO(weights)


try:
    model = load_payload(model_path)
except Exception as e:
    st.sidebar.error(f"Payload Error: {e}. Check weight path.")

# 4. Interface Grid Layout
col_viewport, col_telemetry = st.columns([3, 1])

with col_telemetry:
    st.subheader("📡 Live Telemetry")
    threat_banner = st.empty()
    fps_metric = st.empty()
    dispatch_log = st.empty()

with col_viewport:
    st.subheader("📹 UAV Optical Stream")
    viewport = st.empty()

# 5. Main Detection Loop
if run_patrol:
    # Handle Input Source
    if feed_source == "Live Drone Telemetry (Webcam)":
        cap = cv2.VideoCapture(0)
    else:
        video_file = st.sidebar.file_uploader("Upload Patrol Footage", type=["mp4", "avi"])
        if video_file:
            tfile = tempfile.NamedTemporaryFile(delete=False)
            tfile.write(video_file.read())
            cap = cv2.VideoCapture(tfile.name)
        else:
            st.warning("Please upload a target patrol video file.")
            st.stop()

    prev_time = 0

    while cap.isOpened() and run_patrol:
        ret, frame = cap.read()
        if not ret:
            st.info("Patrol Circuit Completed / Stream Disconnected.")
            break

        # FPS Calculation
        curr_time = time.time()
        fps = 1 / (curr_time - prev_time) if (curr_time - prev_time) > 0 else 30
        prev_time = curr_time

        # YOLO11n Inference
        results = model.predict(frame, conf=conf_thresh, verbose=False)
        annotated_frame = results[0].plot()

        # Check for positive hazard triggers
        detections = results[0].boxes
        if len(detections) > 0:
            top_conf = float(detections.conf[0])
            threat_banner.error(f"🚨 CRITICAL HAZARD DETECTED!\n\n**Confidence:** {top_conf*100:.1f}%")
            dispatch_log.code(
                f"[DISPATCH] Sector A-7 Anomaly\n[TYPE] Combustion/Explosion\n[TIME] {time.strftime('%H:%M:%S')}",
                language="text",
            )
        else:
            threat_banner.success("🛡️ SECTOR CLEAR: Normal Patrol")
            dispatch_log.caption("No localized anomalies registered.")

        # Update Telemetry & Viewport
        fps_metric.metric("Inference Velocity", f"{fps:.1f} FPS")

        # Convert OpenCV BGR to Streamlit RGB
        rgb_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
        viewport.image(rgb_frame, channels="RGB", use_column_width=True)

    cap.release()
else:
    threat_banner.info("🛰️ System Idle. Check 'INITIATE ACTIVE SCAN' to engage UAV sensors.")