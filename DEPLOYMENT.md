# Deploying AeroGuard AI to Streamlit Community Cloud

## Files you need in your GitHub repo (root folder)
- `app.py`
- `requirements.txt`
- `packages.txt` (apt-level libs — OpenCV/ultralytics need these on Cloud)
- `best.pt` (your trained weights)

## Steps
1. **Push to GitHub**
   Create a repo (e.g. `AeroGuard_AI`) and push all four files above to it.
   `best.pt` is ~5.5 MB, so it's fine for a normal git push — GitHub's soft
   limit is 100 MB per file.

2. **Deploy**
   - Go to https://share.streamlit.io
   - Click "New app", pick your repo/branch, and set the main file to `app.py`.
   - Click Deploy. First build takes a few minutes (installing torch/ultralytics).

3. **Set the checkpoint path**
   In the sidebar "Checkpoint Path" field, use `best.pt` (relative path) —
   this matches wherever the repo places the file at deploy time.

## Important limitation: webcam mode won't work on Cloud
Streamlit Community Cloud runs in a headless container with **no camera
access**. `cv2.VideoCapture(0)` will simply fail there. Options:

- For your FYP demo, use **"Pre-Recorded UAV Patrol (MP4)"** mode with
  sample drone footage uploaded through the app — this works perfectly
  on Cloud and is honestly the more reliable choice for live evaluation.
- If you specifically need live webcam detection, run the app **locally**
  (`streamlit run app.py`) or deploy to a VM/VPS where you control camera
  access, instead of Community Cloud.

## Performance note
Community Cloud's free tier gives CPU-only compute (no GPU), so YOLO11n
inference will run at CPU speed. For a nano model this is usually still
several FPS on video files, which is fine for a demo, but don't expect
real-time webcam-grade FPS.

## Local testing before you push
```bash
pip install -r requirements.txt
streamlit run app.py
```
