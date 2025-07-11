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
from torchvision import datasets

# Config
MODEL_PATH = "notebooks/saved_model/vit_face_classifier.pth"
EMBEDDINGS_FILE = "notebooks/saved_embeddings/face_embeddings.npy"
NAMES_FILE = "notebooks/saved_embeddings/face_names.npy"
ATTENDANCE_CSV = "attendance_log.csv"
SIMILARITY_THRESHOLD = 0.65

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
st.title("Smart Attendance System")

# Sidebar for live tracking
st.sidebar.title("Live Dashboard")
prediction_placeholder = st.sidebar.empty()
score_placeholder = st.sidebar.empty()
accuracy_placeholder = st.sidebar.empty()
class_count_placeholder = st.sidebar.empty()

# Prediction, accuracy, and count tracking
prediction_list = []
score_list = []
class_counts = {name: 0 for name in os.listdir("registered_faces")}

option = st.selectbox("Choose an Option", ["Run Attendance", "View Attendance Log", "Add New Person"])

camera_index = 0

if option == "Run Attendance":
    st.info("Click start attendance and align your face in front of the camera.")
    run = st.button("Start Attendance")
    if run:
        embeddings, names = load_registered_embeddings()
        cap = cv2.VideoCapture(camera_index)
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
                class_counts[name] += 1
            else:
                name = "Unknown"
                color = (0, 0, 255)

            # Update the prediction and score
            prediction_list.append(name)
            score_list.append(max_sim)

            # Calculate accuracy
            correct_preds = sum([1 for p, t in zip(prediction_list, names) if p == t])
            accuracy = correct_preds / len(prediction_list) * 100 if len(prediction_list) > 0 else 0

            # Update sidebar with prediction, score, accuracy, and class count
            prediction_placeholder.text(f"Latest Prediction: {name}")
            score_placeholder.text(f"Latest Score: {max_sim:.2f}")
            accuracy_placeholder.text(f"Accuracy: {accuracy:.2f}%")
            class_count_placeholder.text(f"Class Counts: {class_counts}")

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

elif option == "Add New Person":
    name = st.text_input("Enter Name of the Person")
    capture_btn = st.button("Capture & Save Face (30 images)")

    if name and capture_btn:
        st.info(f"Capturing 30 images for {name}... Please click different angles of your face.")
        save_dir = os.path.join("registered_faces", name)
        os.makedirs(save_dir, exist_ok=True)

        cap = cv2.VideoCapture(camera_index)
        captured = 0
        total_images = 30
        stframe = st.empty()

        while captured < total_images:
            ret, frame = cap.read()
            if not ret:
                continue

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            stframe.image(rgb, channels="RGB", caption=f"Capturing Image {captured+1}/{total_images}")

            img_pil = Image.fromarray(rgb)
            img_path = os.path.join(save_dir, f"{captured}.jpg")
            img_pil.save(img_path)
            captured += 1

            cv2.waitKey(500)  # 500 ms pause between captures

        cap.release()
        cv2.destroyAllWindows()
        st.success(f"Images captured and saved for {name}")

        # Re-import if not already
        from torchvision import datasets

        # Regenerate embeddings
        st.info("Updating embeddings...")

        transform_refresh = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])

        dataset = datasets.ImageFolder(root="registered_faces", transform=transform_refresh)
        loader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=False)

        new_embeddings, new_names = [], []

        with torch.no_grad():
            for imgs, labels in loader:
                imgs = imgs.to(device)
                output = model.vit(pixel_values=imgs)
                pooled = output.last_hidden_state[:, 0].cpu().numpy()
                new_embeddings.append(pooled)
                new_names += [dataset.classes[label] for label in labels]

        all_embeddings = np.vstack(new_embeddings)
        all_names = np.array(new_names)

        np.save(EMBEDDINGS_FILE, all_embeddings)
        np.save(NAMES_FILE, all_names)

        st.success("Embeddings updated successfully")
