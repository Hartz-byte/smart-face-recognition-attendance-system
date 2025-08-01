import streamlit as st
import cv2
import numpy as np
from PIL import Image
import requests
import tempfile
import time

API_URL_ROOT = "http://localhost:8000"

st.set_page_config(
    page_title="Smart Attendance System (Client)",
    page_icon="🟢",
    layout="wide"
)

st.title("🟢 Smart Attendance System (API Client)")
st.sidebar.title("Live API Client")

option = st.sidebar.selectbox("Choose Option", [
    "Run System (Camera)", 
    "View Attendance", 
    "Add New Person", 
    "Update Embeddings"
])
api_status = st.sidebar.empty()

def call_predict_api(frame):
    _, img_encoded = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
    files = {'file': ("frame.jpg", img_encoded.tobytes(), 'image/jpeg')}
    try:
        resp = requests.post(f"{API_URL_ROOT}/predict", files=files)
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        return {"error": str(e)}

if option == "Run System (Camera)":
    st.info("Hold 'q' in video window or Stop to quit.")
    camera_index = st.sidebar.number_input("Camera Index", value=0, step=1)
    sample_interval = st.sidebar.slider("Sampling Interval (sec)", 0.1, 2.0, 0.5, 0.1)
    frame_width = st.sidebar.selectbox("Frame Width", [640, 320, 160], index=1)
    frame_height = st.sidebar.selectbox("Frame Height", [480, 240, 120], index=1)
    stframe = st.empty()
    run = st.button("Start Camera", type="primary")
    stop = st.button("Stop")
    if run:
        cap = cv2.VideoCapture(int(camera_index))
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                st.warning("Failed to read from camera.")
                break
            frame_resized = cv2.resize(frame, (frame_width, frame_height))
            result = call_predict_api(frame_resized)
            frame_disp = frame_resized.copy()
            if result.get("error"):
                label = f"API ERROR: {result['error']}"
                color = (0, 0, 255)
            elif result.get('spoof', False):
                label = f"🚨 SPOOF DETECTED! Score: {result.get('spoof_score', 0):.2f}"
                color = (0, 0, 255)
            else:
                name = result.get('name', 'Unknown')
                emotion = result.get('emotion', 'Unknown')
                spoof_score = result.get('spoof_score', 0)
                label = f"{name} | {emotion} | Spoof: {spoof_score:.2f}"
                color = (0, 255, 0)
            cv2.putText(frame_disp, label, (8, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.85, color, 2)
            stframe.image(frame_disp[..., ::-1], channels="RGB", caption=label)
            if stop or cv2.waitKey(1) & 0xFF == ord('q'):
                break
            time.sleep(sample_interval)
        cap.release()
        cv2.destroyAllWindows()

elif option == "View Attendance":
    st.subheader("Attendance Log")
    try:
        resp = requests.get(f"{API_URL_ROOT}/attendance")
        resp.raise_for_status()
        data = resp.json()
        if data.get("attendance"):
            st.dataframe(data["attendance"])
        else:
            st.info(data.get("message", "No attendance data."))
    except Exception as e:
        st.error(f"Failed to fetch attendance: {e}")

elif option == "Add New Person":
    st.subheader("Register New Person")
    name = st.text_input("Person's Name")
    uploaded_files = st.file_uploader(
        "Upload one or more clear face images (JPEG/PNG)", 
        type=['jpg', 'jpeg', 'png'], 
        accept_multiple_files=True
    )
    if st.button("Register") and name and uploaded_files:
        files = [
            ("files", (file.name, file.read(), file.type))
            for file in uploaded_files
        ]
        data = {"name": name.strip()}
        try:
            resp = requests.post(f"{API_URL_ROOT}/add-person", data=data, files=files)
            resp.raise_for_status()
            result = resp.json()
            st.success(result["message"])
        except Exception as e:
            st.error(f"Registration failed: {e}")

elif option == "Update Embeddings":
    st.subheader("Update Embeddings")
    if st.button("Regenerate Now", type="primary"):
        try:
            resp = requests.post(f"{API_URL_ROOT}/update-embeddings")
            resp.raise_for_status()
            result = resp.json()
            st.success(result["message"])
        except Exception as e:
            st.error(f"Embedding update failed: {e}")
