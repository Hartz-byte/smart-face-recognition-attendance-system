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
from transformers import ViTModel

# Config
MODEL_PATH = "notebooks/saved_model/vit_face_classifier.pth"
EMBEDDINGS_FILE = "notebooks/saved_embeddings/face_embeddings.npy"
NAMES_FILE = "notebooks/saved_embeddings/face_names.npy"
ATTENDANCE_CSV = "attendance_log.csv"
SIMILARITY_THRESHOLD = 0.7

# Device & transform
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# Define model
class FaceClassifier(nn.Module):
    def __init__(self, vit, num_classes):
        super().__init__()
        self.vit = vit
        self.classifier = nn.Linear(vit.config.hidden_size, num_classes)

    def forward(self, x):
        outputs = self.vit(pixel_values=x)
        pooled = outputs.last_hidden_state[:, 0]
        return self.classifier(pooled)

vit = ViTModel.from_pretrained("google/vit-base-patch16-224")
model = FaceClassifier(vit, num_classes=3)
checkpoint = torch.load(MODEL_PATH, map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
model.to(device)
model.eval()

# Utils
def get_embedding(image):
    img_tensor = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model.vit(pixel_values=img_tensor)
        return output.last_hidden_state[:, 0].cpu().numpy()[0]

def load_registered_embeddings():
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

# StreamLit UI
st.title("Smart Attendance System with ViT")
option = st.selectbox("Choose an Option", ["Run Attendance", "View Attendance Log"])

if option == "Run Attendance":
    st.info("Click start attendance and align your face in front of the camera.")
    run = st.button("Start Attendance")
    if run:
        embeddings, names = load_registered_embeddings()
        cap = cv2.VideoCapture(0)
        stframe = st.empty()

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            face_img = Image.fromarray(rgb)
            emb = get_embedding(face_img)
            sims = cosine_similarity([emb], embeddings)[0]
            max_idx = np.argmax(sims)
            max_sim = sims[max_idx]

            if max_sim > SIMILARITY_THRESHOLD:
                name = names[max_idx]
                mark_attendance(name)
                color = (0, 180, 0)
            else:
                name = "Unknown"
                color = (0, 0, 255)

            cv2.putText(frame, f"{name} ({max_sim:.2f})", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
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
