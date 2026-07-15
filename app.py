"""
AeroGuard AI - UAV Early Warning C2 Dashboard
Author: Irum Nigar Wazir
"""

import tempfile
import time
import cv2
import numpy as np
from PIL import Image
import streamlit as st
from ultralytics import YOLO
import av
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration

# ----------------------------------------------------------------------------
# 1. Page Configuration
# ----------------------------------------------------------------------------
st.set_page_config(page_title="AeroGuard AI | UAV Command", page_icon="🚁", layout="wide")

st.title("🚁 AeroGuard AI: Real-Time UAV Hazard Dispatch")
st.markdown("**Active Payload:** YOLO11n (FP16) | **Status:** `MONITORING`")
st.divider()

# ----------------------------------------------------------------------------
# 2. Sidebar & Model Loading
# ----------------------------------------------------------------------------
st.sidebar.header("⚙️ Flight Controller Settings")
model_path = st.sidebar.text_input("Checkpoint Path", "best.pt")
conf_thresh = st.sidebar.slider("Confidence Sensitivity", 0.10, 1.00, 0.45, 0.05)
feed_source = st.sidebar.radio(
    "Optical Input Source",
    ["Snapshot Capture (Camera)", "Live Video Stream (WebRTC)", "Pre-Recorded UAV Patrol (MP4)"],
)

@st.cache_resource(show_spinner="Loading model weights…")
def load_payload(weights_path: str):
    return YOLO(weights_path)

model = load_payload(model_path)

# WebRTC Config
RTC_CONFIGURATION = RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]})

# ----------------------------------------------------------------------------
# 3. Inference Helper
# ----------------------------------------------------------------------------
def run_inference_and_display(frame_bgr, fps=None, banner=None, log=None, viewport=None, metric=None):
    results = model.predict(frame_bgr, conf=conf_thresh, verbose=False)
    annotated_frame = results[0].plot()

    detections = results[0].boxes
    if len(detections) > 0:
        label = model.names.get(int(detections.cls[0]), "Unknown")
        banner.error(f"🚨 HAZARD: {label} ({float(detections.conf[0])*100:.1f}%)")
        log.code(f"[DISPATCH] {label} detected at {time.strftime('%H:%M:%S')}")
    else:
        banner.success("🛡️ SECTOR CLEAR")
    
    if metric and fps:
        metric.metric("Inference Velocity", f"{fps:.1f} FPS")
    
    return cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)

# ----------------------------------------------------------------------------
# 4. Main Logic
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

# Option 1: Snapshot
if feed_source == "Snapshot Capture (Camera)":
    photo = st.camera_input("📸 Capture Frame")
    if photo:
        frame = cv2.cvtColor(np.array(Image.open(photo)), cv2.COLOR_RGB2BGR)
        res_img = run_inference_and_display(frame, banner=threat_banner, log=dispatch_log)
        viewport.image(res_img, use_container_width=True)

# Option 2: Live WebRTC
elif feed_source == "Live Video Stream (WebRTC)":
    def video_frame_callback(frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        # Note: We can't update Streamlit widgets from here easily
        results = model.predict(img, conf=conf_thresh, verbose=False)
        return av.VideoFrame.from_ndarray(results[0].plot(), format="bgr24")

    webrtc_streamer(
        key="uav-live",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration=RTC_CONFIGURATION,
        video_frame_callback=video_frame_callback,
    )

# Option 3: Pre-Recorded
else:
    video_file = st.sidebar.file_uploader("Upload Patrol Footage", type=["mp4", "avi"])
    if video_file:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(video_file.read())
        cap = cv2.VideoCapture(tfile.name)
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            res_img = run_inference_and_display(frame, fps=30, banner=threat_banner, log=dispatch_log, viewport=viewport, metric=fps_metric)
            viewport.image(res_img, use_container_width=True)
        cap.release()
