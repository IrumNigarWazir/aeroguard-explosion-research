"""
AeroGuard AI - UAV Early Warning C2 Dashboard
Author: Irum Nigar Wazir
FYP-II: Real-Time Aerial Hazard Detection
"""

import tempfile
import time

import cv2
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
    ["Live Drone Telemetry (Webcam)", "Pre-Recorded UAV Patrol (MP4)"],
)

# Give the checkbox a stable key so we can poll st.session_state for a live
# stop signal from inside the detection loop below.
RUN_KEY = "run_patrol"
run_patrol = st.sidebar.checkbox("▶ INITIATE ACTIVE SCAN", value=False, key=RUN_KEY)


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

# ----------------------------------------------------------------------------
# 5. Main Detection Loop
# ----------------------------------------------------------------------------
if run_patrol and model is not None:
    cap = None
    try:
        if feed_source == "Live Drone Telemetry (Webcam)":
            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                st.error(
                    "No webcam detected. Note: Streamlit Community Cloud has no "
                    "camera access — webcam mode only works when run locally."
                )
                st.stop()
        else:
            video_file = st.sidebar.file_uploader(
                "Upload Patrol Footage", type=["mp4", "avi", "mov"]
            )
            if video_file:
                tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
                tfile.write(video_file.read())
                tfile.flush()
                cap = cv2.VideoCapture(tfile.name)
            else:
                st.warning("Please upload a target patrol video file.")
                st.stop()

        prev_time = time.time()

        while cap.isOpened():
            # Live stop signal: reflects the checkbox even mid-loop, since
            # Streamlit updates session_state as soon as the widget changes.
            if not st.session_state.get(RUN_KEY, False):
                st.info("Scan halted by operator.")
                break

            ret, frame = cap.read()
            if not ret:
                st.info("Patrol Circuit Completed / Stream Disconnected.")
                break

            # FPS calculation
            curr_time = time.time()
            elapsed = curr_time - prev_time
            fps = 1 / elapsed if elapsed > 0 else 0.0
            prev_time = curr_time

            # YOLO inference
            results = model.predict(frame, conf=conf_thresh, verbose=False)
            annotated_frame = results[0].plot()

            # Hazard trigger check
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

            fps_metric.metric("Inference Velocity", f"{fps:.1f} FPS")

            rgb_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
            viewport.image(rgb_frame, channels="RGB", use_container_width=True)

    finally:
        if cap is not None:
            cap.release()

elif run_patrol and model is None:
    st.error("Cannot start scan — model failed to load. Check the checkpoint path in the sidebar.")
else:
    threat_banner.info("🛰️ System Idle. Check 'INITIATE ACTIVE SCAN' to engage UAV sensors.")
