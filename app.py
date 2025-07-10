# Imports
import os
import torch
from torch import nn
from torchvision import transforms
from PIL import Image
import cv2
import numpy as np
import pandas as pd
from datetime import datetime
import streamlit as st
from sklearn.metrics.pairwise import cosine_similarity
from facenet_pytorch import MTCNN
from transformers import ViTModel
from collections import deque, Counter

# Config
MODEL_PATH = "notebooks/saved_model/vit_face_classifier.pth"
EMBEDDINGS_FILE = "notebooks/saved_embeddings/face_embeddings.npy"
NAMES_FILE = "notebooks/saved_embeddings/face_names.npy"
ATTENDANCE_CSV = "attendance_log.csv"
SIMILARITY_THRESHOLD = 0.6
SMOOTHING_WINDOW = 5

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Face detector: margin=40 captures more context
mtcnn = MTCNN(image_size=224, margin=40)

# Transform: match ViT pretrained normalization
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5])
])

# Model definition (same as training)
class FaceClassifier(nn.Module):
    def __init__(self, vit, num_classes):
        super().__init__()
        self.vit = vit
        self.classifier = nn.Linear(vit.config.hidden_size, num_classes)
    def forward(self, x):
        outputs = self.vit(pixel_values=x)
        pooled = outputs.last_hidden_state[:, 0]
        return self.classifier(pooled)

# Load model
vit = ViTModel.from_pretrained("google/vit-base-patch16-224")
model = FaceClassifier(vit, num_classes=3)
checkpoint = torch.load(MODEL_PATH, map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
model = model.to(device)
model.eval()

# Utils
def get_embedding(image):
    face = mtcnn(image)
    if face is None:
        return None

    face = transforms.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5])(face)
    face = face.unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model.vit(pixel_values=face)
        emb = outputs.last_hidden_state[:,0]
    return emb.cpu().numpy()[0]


def load_registered_embeddings():
    embeddings = np.load(EMBEDDINGS_FILE)
    names = np.load(NAMES_FILE)
    return embeddings, names

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

# Streamlit UI
st.title("📸 Smart Attendance System with ViT")

option = st.selectbox("Choose an Option", ["Run Attendance", "View Attendance Log"])

if option == "Run Attendance":
    st.info("Click start and align your face in front of the camera. Press 'q' to stop.")
    run = st.button("Start Attendance")
    if run:
        reg_embeddings, reg_names = load_registered_embeddings()
        cap = cv2.VideoCapture(0)
        stframe = st.empty()

        # for smoothing
        recent_predictions = deque(maxlen=SMOOTHING_WINDOW)

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            face_img = Image.fromarray(rgb)

            emb = get_embedding(face_img)
            if emb is not None:
                sims = cosine_similarity([emb], reg_embeddings)[0]
                max_idx = np.argmax(sims)
                max_sim = sims[max_idx]
                predicted_name = reg_names[max_idx]

                st.text(f"Similarity: {max_sim:.2f} → {predicted_name}")

                if max_sim > SIMILARITY_THRESHOLD:
                    recent_predictions.append(predicted_name)
                else:
                    recent_predictions.append("Unknown")

                # majority voting
                smoothed_name, _ = Counter(recent_predictions).most_common(1)[0]

                if smoothed_name != "Unknown":
                    mark_attendance(smoothed_name)
                    color = (0, 180, 0)
                else:
                    color = (0, 0, 255)

                label_text = f"{smoothed_name} ({max_sim:.2f})"
            else:
                label_text = "No face"
                color = (255, 0, 0)

            cv2.putText(frame, label_text, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
            stframe.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

elif option == "View Attendance Log":
    if os.path.exists(ATTENDANCE_CSV):
        df = pd.read_csv(ATTENDANCE_CSV)
        st.dataframe(df)
    else:
        st.warning("No attendance log found yet.")
