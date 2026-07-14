Markdown
# 🚁 AeroGuard AI: Real-Time UAV Hazard & Explosion Detection

**AeroGuard AI** is a Command and Control (C2) dashboard designed for Unmanned Aerial Vehicles (UAVs) to provide early warning detection of localized anomalies, specifically focusing on combustion and explosions. 

Developed as a Final Year Project at the Agriculture University Peshawar, this system leverages advanced computer vision and edge technologies to process live optical streams and telemetry, ensuring rapid hazard response in wide-area sectors.

---

## 🌟 Key Features

*   **Real-Time Active Scanning:** Processes live drone telemetry (webcam) or pre-recorded UAV patrol footage (MP4/AVI).
*   **Lightweight Edge Inference:** Powered by YOLOv11n (FP16), specifically chosen for high-speed, lightweight aerial detection suitable for edge devices.
*   **Military-Grade C2 Interface:** A responsive dashboard built with Streamlit featuring live confidence scoring, inference velocity (FPS) tracking, and automated threat banners.
*   **Automated Dispatch Logging:** Registers localized anomalies with timestamped logs for rapid sector dispatch.

---

## 🔬 Methodology & Research

This research project focuses on pushing the boundaries of real-time computer vision in aerial robotics:

*   **Dataset Creation:** The model was rigorously trained using custom drone dataset imagery to accurately recognize explosion and combustion signatures from aerial perspectives.
*   **Model Architecture:** Utilizes the YOLO (You Only Look Once) real-time object detection framework, specifically targeting the `nano` configuration (YOLOv11n) to balance high detection accuracy with the low computational overhead required for drone deployments.

---

## 🛠️ Tech Stack

*   **Deep Learning:** PyTorch, Ultralytics (YOLO)
*   **Computer Vision:** OpenCV (`opencv-python`)
*   **Frontend / Dashboard:** Streamlit
*   **Data Processing:** NumPy, Pillow

---

## 🚀 Installation & Local Deployment

To run the AeroGuard AI dashboard locally on your machine, follow these steps:

### 1. Clone the Repository
```
git clone [https://github.com/IrumNigarWazir/aeroguard-explosion-research.git](https://github.com/IrumNigarWazir/aeroguard-explosion-research.git)
cd aeroguard-explosion-research```
### 2. Install Dependencies
Ensure you have Python installed, then run:


`pip install -r requirements.txt`
### 3. Add Model Weights
Place your custom-trained YOLOv11n model file in the root directory and ensure it is named best.pt (or update the model_path variable in the sidebar of the application).

### 4. Launch the Dashboard

`streamlit run app.py`
## 🌐 Streamlit Community Cloud Deployment
This project is optimized for deployment on Streamlit Community Cloud.

Log in to Streamlit Cloud.

Click New app.

Paste the repository URL: https://github.com/IrumNigarWazir/aeroguard-explosion-research

Set the Main file path to app.py.

Click Deploy.

### 👥 Contributors
Irum Nigar Wazir - Lead Developer & Researcher

Sadia - Collaborator / Project Partner

📝 License
This project is submitted in partial fulfillment of the requirements for the final year engineering project. Educational and research use is encouraged.
