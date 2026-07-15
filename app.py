"""
AeroGuard AI - UAV Early Warning C2 Dashboard
Author: Irum Nigar Wazir
FYP-II: Real-Time Aerial Hazard Detection
"""

import tempfile
import time

import cv2
import numpy as np
from PIL import Image
import streamlit as st
from ultralytics import YOLO

# ----------------------------------------------------------------------------
# 1. Page Configuration
# ----------------------------------------------------------------------------
st.set_page_config(
    page_title="AeroGuard AI | UAV Command", page_icon="🚁", layout="wide"
)

st.title("🚁 AeroGuard AI: Real-Time UAV Hazard Dispatch")
st.markdown(
    "**Active Payload:** YOLO11n (FP16) | **Target Sector:** Wide-Area (40+ Kanal) | "
    "**Status:** `MONITORING`"
)
st.divider()

# ----------------------------------------------------------------------------
# 2. Sidebar Controls
# ----------------------------------------------------------------------------
st.sidebar.header("⚙️ Flight Controller Settings")
model_path = st.sidebar.text_input("Checkpoint Path", "best.pt")
conf_thresh = st.sidebar.slider("Confidence Sensitivity", 0.10, 1.00, 0.45, 0.05)
feed_source = st.sidebar.radio(
    "Optical Input Source",
    ["Snapshot Capture (Camera)", "Pre-Recorded UAV Patrol (MP4)"],
)


# ----------------------------------------------------------------------------
# 3. Load YOLO model (cached so it isn't reloaded every rerun)
# ----------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading model weights…")
def load_payload(weights_path: str):
    return YOLO(weights_path)


model = None
try:
    model = load_payload(model_path)
    st.sidebar.success(f"Payload loaded: `{model_path}`")
except Exception as e:
    st.sidebar.error(f"Payload Error: {e}\nCheck the weights path.")

# ----------------------------------------------------------------------------
# 4. Interface Grid Layout
# ----------------------------------------------------------------------------
col_viewport, col_telemetry = st.columns([3, 1])

with col_telemetry:
    st.subheader("📡 Live Telemetry")
    threat_banner = st.empty()
    fps_metric = st.empty()
    dispatch_log = st.empty()

with col_viewport:
    st.subheader("📹 UAV Optical Stream")
    viewport = st.empty()


def run_inference_and_display(frame_bgr, fps=None):
    """Run YOLO on a single BGR frame and update the dashboard widgets."""
    results = model.predict(frame_bgr, conf=conf_thresh, verbose=False)
    annotated_frame = results[0].plot()

    detections = results[0].boxes
    if len(detections) > 0:
        top_conf = float(detections.conf[0])
        top_cls_id = int(detections.cls[0])
        label = model.names.get(top_cls_id, "Unknown") if hasattr(model, "names") else "Unknown"

        threat_banner.error(
            f"🚨 CRITICAL HAZARD DETECTED!\n\n"
            f"**Class:** {label}  |  **Confidence:** {top_conf * 100:.1f}%"
        )
        dispatch_log.code(
            f"[DISPATCH] Sector A-7 Anomaly\n"
            f"[TYPE] {label}\n"
            f"[TIME] {time.strftime('%H:%M:%S')}",
            language="text",
        )
    else:
        threat_banner.success("🛡️ SECTOR CLEAR: Normal Patrol")
        dispatch_log.caption("No localized anomalies registered.")

    if fps is not None:
        fps_metric.metric("Inference Velocity", f"{fps:.1f} FPS")
    else:
        fps_metric.metric("Mode", "Single Snapshot")

    rgb_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
    viewport.image(rgb_frame, channels="RGB", use_container_width=True)


# ----------------------------------------------------------------------------
# 5. Input Handling
# ----------------------------------------------------------------------------
if model is None:
    threat_banner.info("🛰️ Awaiting valid payload (model) before sensors can engage.")

elif feed_source == "Snapshot Capture (Camera)":
    st.sidebar.caption("Uses your device's camera via the browser — works locally and on Cloud.")
    photo = st.camera_input("📸 Capture UAV Optical Frame")

    if photo is not None:
        image = Image.open(photo).convert("RGB")
        frame_bgr = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        run_inference_and_display(frame_bgr)
    else:
        threat_banner.info("🛰️ System Idle. Capture a frame above to run a hazard scan.")

else:  # Pre-Recorded UAV Patrol (MP4)
    run_patrol = st.sidebar.checkbox("▶ INITIATE ACTIVE SCAN", value=False, key="run_patrol")
    video_file = st.sidebar.file_uploader("Upload Patrol Footage", type=["mp4", "avi", "mov"])

    if run_patrol:
        if not video_file:
            st.warning("Please upload a target patrol video file.")
            st.stop()

        cap = None
        try:
            tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
            tfile.write(video_file.read())
            tfile.flush()
            cap = cv2.VideoCapture(tfile.name)

            prev_time = time.time()

            while cap.isOpened():
                if not st.session_state.get("run_patrol", False):
                    st.info("Scan halted by operator.")
                    break

                ret, frame = cap.read()
                if not ret:
                    st.info("Patrol Circuit Completed / Stream Disconnected.")
                    break

                curr_time = time.time()
                elapsed = curr_time - prev_time
                fps = 1 / elapsed if elapsed > 0 else 0.0
                prev_time = curr_time

                run_inference_and_display(frame, fps=fps)
        finally:
            if cap is not None:
                cap.release()
    else:
        threat_banner.info("🛰️ System Idle. Upload footage and check 'INITIATE ACTIVE SCAN' to engage.")
