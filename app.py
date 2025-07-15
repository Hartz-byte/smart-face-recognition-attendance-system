import os
import cv2
import torch
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
from datetime import datetime
from torchvision import transforms, datasets
from sklearn.metrics.pairwise import cosine_similarity

from face_model import load_model
from emotion_model import decode_emotion

# Config
MODEL_PATH = "./notebooks/saved_model/triple_head_vit.pth"
EMBEDDINGS_FILE = "./notebooks/saved_embeddings/face_embeddings.npy"
NAMES_FILE = "./notebooks/saved_embeddings/face_names.npy"
ATTENDANCE_CSV = "./attendance_log.csv"
REGISTERED_DIR = "./registered_faces"
SIMILARITY_THRESHOLD = 0.60
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# Load model (triple head)
model, face_class_map, emotion_class_map = load_model(MODEL_PATH, device)

def get_all_predictions(image):
    tensor = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        face_logits, emotion_logits, spoof_logits = model(tensor)
        features = model.vit(pixel_values=tensor).last_hidden_state[:, 0]
        spoof_prob = torch.sigmoid(spoof_logits).item()
        return features[0].cpu().numpy(), emotion_logits[0].cpu(), spoof_prob

def load_embeddings():
    return np.load(EMBEDDINGS_FILE), np.load(NAMES_FILE)

def mark_attendance(name):
    now = datetime.now()
    date_str = now.strftime('%Y-%m-%d')
    time_str = now.strftime('%H:%M:%S')
    if os.path.exists(ATTENDANCE_CSV):
        df = pd.read_csv(ATTENDANCE_CSV)
        if ((df['Name'] == name) & (df['Date'] == date_str)).any():
            return
    else:
        df = pd.DataFrame(columns=['Name', 'Date', 'Time'])
    df = pd.concat([df, pd.DataFrame([{'Name': name, 'Date': date_str, 'Time': time_str}])], ignore_index=True)
    df.to_csv(ATTENDANCE_CSV, index=False)

# UI Setup
st.title("Smart Attendance System")
st.sidebar.title("Live Predictions")
pred_placeholder = st.sidebar.empty()
emotion_placeholder = st.sidebar.empty()
spoof_placeholder = st.sidebar.empty()

option = st.selectbox("Choose Option", ["Run System", "View Attendance", "Add New Person", "Update Embeddings"])
camera_index = 1

if option == "Run System":
    run = st.button("Start Camera")
    if run:
        embeddings, names = load_embeddings()
        cap = cv2.VideoCapture(camera_index)
        stframe = st.empty()

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(rgb)
            emb, emotion_logits, spoof_score = get_all_predictions(pil_img)

            # Spoof Check
            if spoof_score > 0.5:
                text = "Spoof Detected!"
                color = (0, 0, 255)
                cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
                spoof_placeholder.warning(f"⚠️ Spoof Detected! ({spoof_score:.2f})")
                stframe.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                continue

            # Face recognition
            sims = cosine_similarity([emb], embeddings)[0]
            max_idx = np.argmax(sims)
            max_sim = sims[max_idx]

            if max_sim > SIMILARITY_THRESHOLD:
                name = names[max_idx]
                mark_attendance(name)
                color = (0, 255, 0)
            else:
                name = "Unknown"
                color = (255, 0, 0)

            # Emotion
            emotion, emotion_score = decode_emotion(emotion_logits)

            # Display
            text = f"{name} | {emotion} ({emotion_score:.2f})"
            cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

            pred_placeholder.text(f"Name: {name} | Score: {max_sim:.2f}")
            emotion_placeholder.text(f"Emotion: {emotion} ({emotion_score:.2f})")
            spoof_placeholder.success(f"Spoof Score: {spoof_score:.2f} ✅ Real")
            stframe.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

elif option == "View Attendance":
    if os.path.exists(ATTENDANCE_CSV):
        df = pd.read_csv(ATTENDANCE_CSV)
        st.dataframe(df)
    else:
        st.warning("No attendance log available.")

elif option == "Add New Person":
    name = st.text_input("Enter Name of the Person")
    capture_btn = st.button("Capture & Save Face (30 images)")

    if name and capture_btn:
        st.info(f"Capturing 30 images for {name}...")
        save_dir = os.path.join(REGISTERED_DIR, name)
        os.makedirs(save_dir, exist_ok=True)

        cap = cv2.VideoCapture(camera_index)
        captured = 0
        total_images = 30
        stframe = st.empty()

        existing_files = os.listdir(save_dir)
        existing_indices = [int(f.split('.')[0]) for f in existing_files if f.endswith('.jpg') and f.split('.')[0].isdigit()]
        next_index = max(existing_indices) + 1 if existing_indices else 0

        while captured < total_images:
            ret, frame = cap.read()
            if not ret:
                continue
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            stframe.image(rgb, channels="RGB", caption=f"Image {captured+1}/{total_images}")
            img_pil = Image.fromarray(rgb)

            img_pil.save(os.path.join(save_dir, f"{next_index}.jpg"))
            next_index += 1
            captured += 1
            cv2.waitKey(500)

        cap.release()
        cv2.destroyAllWindows()
        st.success(f"Images captured for {name}")
        st.warning("Click 'Update Embeddings' to reflect changes.")

elif option == "Update Embeddings":
    if st.button("Regenerate Now"):
        from torch.utils.data import DataLoader

        st.info("Updating embeddings...")
        dataset = datasets.ImageFolder(root=REGISTERED_DIR, transform=transform)
        loader = DataLoader(dataset, batch_size=4)

        all_embeddings, all_names = [], []
        with torch.no_grad():
            for imgs, labels in loader:
                imgs = imgs.to(device)
                features = model.vit(pixel_values=imgs).last_hidden_state[:, 0]
                all_embeddings.append(features.cpu().numpy())
                all_names += [dataset.classes[label] for label in labels]

        np.save(EMBEDDINGS_FILE, np.vstack(all_embeddings))
        np.save(NAMES_FILE, np.array(all_names))
        st.success("Embeddings updated.")
