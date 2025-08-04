import streamlit as st
import cv2
import numpy as np
import requests
import os
import json
from datetime import datetime
import time

API_URL_ROOT = "http://localhost:8000"
FEEDBACK_DIR = "./feedback_frames"
FEEDBACK_LOG = os.path.join(FEEDBACK_DIR, "feedback_log.json")
EMOTION_CLASSES = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]
os.makedirs(FEEDBACK_DIR, exist_ok=True)

def call_predict_api(frame):
    _, img_encoded = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
    files = {'file': ("frame.jpg", img_encoded.tobytes(), 'image/jpeg')}
    try:
        resp = requests.post(f"{API_URL_ROOT}/predict", files=files)
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        return {"error": str(e)}

def append_feedback(frame_np, pred_name, corrected_name, pred_emotion, corrected_emotion):
    ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    img_path = os.path.join(FEEDBACK_DIR, f"{ts}.jpg")
    cv2.imwrite(img_path, frame_np)
    feedback_entry = {
        "timestamp": ts,
        "image": img_path,
        "predicted_name": pred_name,
        "corrected_name": corrected_name,
        "predicted_emotion": pred_emotion,
        "corrected_emotion": corrected_emotion
    }
    if os.path.exists(FEEDBACK_LOG):
        with open(FEEDBACK_LOG, "r") as f:
            log = json.load(f)
    else:
        log = []
    log.append(feedback_entry)
    with open(FEEDBACK_LOG, "w") as f:
        json.dump(log, f, indent=2)

if "camera_running" not in st.session_state:
    st.session_state.camera_running = False
if "feedback_mode" not in st.session_state:
    st.session_state.feedback_mode = False
if "last_frame" not in st.session_state:
    st.session_state.last_frame = None
    st.session_state.last_pred_name = None
    st.session_state.last_pred_emotion = None

st.set_page_config(page_title="Smart Attendance System (Client)", page_icon="🟢", layout="wide")
st.title("🟢 Smart Attendance System (API Client)")
st.sidebar.title("Live API Client")

option = st.sidebar.selectbox("Choose Option", [
    "Run System (Camera)", "View Attendance", "Add New Person", "Update Embeddings"
])

if option == "Run System (Camera)":
    sample_interval = st.sidebar.slider("Sampling Interval (sec)", 0.1, 2.0, 0.5, 0.1)
    frame_width = st.sidebar.selectbox("Frame Width", [640, 320, 160], index=0)
    frame_height = st.sidebar.selectbox("Frame Height", [480, 240, 120], index=0)
    cam_index = st.sidebar.number_input("Camera Index", value=0, step=1)
    stframe = st.empty()
    run = st.button("Start Camera", type="primary", key="start_cam")
    stop = st.button("Stop", key="stop_cam")
    feedback_btn = st.button("Capture for Feedback", key="feedback_btn")

    # Start the camera loop
    if run and not st.session_state.camera_running:
        st.session_state.camera_running = True
        st.session_state.feedback_mode = False

    if stop:
        st.session_state.camera_running = False
        st.session_state.feedback_mode = False

    cap = None
    if st.session_state.camera_running and not st.session_state.feedback_mode:
        cap = cv2.VideoCapture(int(cam_index))
        while st.session_state.camera_running and not st.session_state.feedback_mode:
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
                pred_name, pred_emotion = "Unknown", EMOTION_CLASSES[-1]
            elif result.get('spoof', False):
                label = f"🚨 SPOOF DETECTED! Score: {result.get('spoof_score', 0):.2f}"
                color = (0, 0, 255)
                pred_name, pred_emotion = "Unknown", EMOTION_CLASSES[-1]
            else:
                pred_name = result.get('name', 'Unknown')
                pred_emotion = result.get('emotion', EMOTION_CLASSES[-1])
                spoof_score = result.get('spoof_score', 0)
                label = f"{pred_name} | {pred_emotion} | Spoof: {spoof_score:.2f}"
                color = (0, 255, 0)
            cv2.putText(frame_disp, label, (8, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.85, color, 2)
            stframe.image(frame_disp[..., ::-1], channels="RGB", caption=label)

            # Save this frame in session_state in case feedback is given
            st.session_state.last_frame = frame_resized.copy()
            st.session_state.last_pred_name = pred_name
            st.session_state.last_pred_emotion = pred_emotion

            # Allow Streamlit events to update
            time.sleep(sample_interval)

            # If the "Capture for Feedback" button has been clicked, show feedback UI
            if feedback_btn:
                st.session_state.feedback_mode = True
                st.session_state.camera_running = False

            if st.session_state.feedback_mode or stop:
                break
        if cap and cap.isOpened():
            cap.release()
        cv2.destroyAllWindows()

    # Feedback form
    if st.session_state.feedback_mode and st.session_state.last_frame is not None:
        with st.form(key="feedback_form"):
            st.write("#### Prediction Feedback (for captured frame)")
            corrected_name = st.text_input("Correct Name:", value=st.session_state.last_pred_name)
            corrected_emotion = st.selectbox(
                "Correct Emotion:", EMOTION_CLASSES,
                index=EMOTION_CLASSES.index(st.session_state.last_pred_emotion)
                if st.session_state.last_pred_emotion in EMOTION_CLASSES else 0
            )
            submit_feedback = st.form_submit_button("Send Feedback for This Frame")
            if submit_feedback:
                append_feedback(
                    st.session_state.last_frame,
                    st.session_state.last_pred_name,
                    corrected_name,
                    st.session_state.last_pred_emotion,
                    corrected_emotion
                )
                st.success("✅ Feedback submitted!")
                # Reset for next round
                st.session_state.feedback_mode = False
                st.session_state.last_frame = None
                st.session_state.last_pred_name = None
                st.session_state.last_pred_emotion = None

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
