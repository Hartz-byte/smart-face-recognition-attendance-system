# Smart Multi-Task Face Recognition System
A real-world, multi-task deep learning application that performs:
- Face Recognition (Attendance System)
- Emotion Detection
- Spoof Detection (Liveness Check)

This is a portfolio-grade personal project, showcasing an end-to-end AI system with:
- A custom-built dataset
- A multi-head ViT-based model
- A Streamlit-powered UI
- Real-time camera integration

Despite dataset limitations (especially small spoof/emotion datasets), the system performs impressively and demonstrates strong full-stack ML engineering capabilities.

---

## Features
- Face Recognition
  - Based on the cosine similarity of ViT-extracted features, supports adding new persons live via webcam.
- Emotion Detection
  - Predicts emotion using a dedicated model head.
- Spoof Detection
  - Uses a binary classification head to detect spoof attacks.
- Attendance Logging
  - CSV-based attendance with duplicate filtering per day.
- Embeddings Management
  - Add/remove/update embeddings via simple UI options.

---

# Architecture Overview
## Model Architecture
Implemented using the Vision Transformer (vit-base-patch16-224) from Hugging Face:
- DualHeadViT
  - Face Classification
  - Emotion Classification

- TripleHeadViT
  - Face Classification
  - Emotion Classification
  - Spoof Detection (Binary)

```bash
class DualHeadViT(nn.Module):
    ...
    def forward(self, x):
        features = self.vit(pixel_values=x).last_hidden_state[:, 0]
        return self.face_head(features), self.emotion_head(features)
```

```bash
class TripleHeadViT(nn.Module):
    ...
    def forward(self, x):
        features = self.vit(pixel_values=x).last_hidden_state[:, 0]
        return self.face_head(features), self.emotion_head(features), self.spoof_head(features)
```

---

## Components
- face_model.py – Model definition and loader.
- app.py – Streamlit UI & main app.
- emotion_model.py – Emotion decoding logic.
- Training Notebooks:
  - train_face_head.ipynb
  - train_emotion_head.ipynb
  - train_spoof_head.ipynb
  - train_dual_head_model.ipynb
  - train_triple_head.ipynb

---

## Dataset Creation
This project uses hand-crafted datasets for face, spoof, and emotion recognition:
- Registered Faces
  - Images captured using webcam of my family members (30+ images/person).
  - Stored in folder structure under registered_faces/.

- Emotion Dataset
  - A small custom dataset created manually.
  - Performance is limited due to small size – noticeable overfitting.

- Spoof Dataset
  - Captured using spoof images (photos from the screen, printed faces).
  - Trained as a binary classifier (real vs. spoof).
  - Small dataset, but sufficient for proof-of-concept.
> Note: Due to the small dataset sizes, model overfitting is expected; however, the application still performs well in practice.

---

## How It Works
1. Start the System
    - Select “Run System” from the Streamlit sidebar.
    - Start webcam capture.
    - Live face/emotion/spoof predictions are shown in real-time.
    - Attendance is auto-marked if the person is recognized and real.

2. Add New Person
    - Enter name and capture 30 images via webcam.
    - Stored locally under registered_faces/.

3. Update Embeddings
    - Regenerate facial embeddings using the ViT backbone.
    - Saves .npy files for use in recognition.

4. View Attendance
    - Display attendance logs stored in attendance_log.csv.

---

## Challenges Faced
| Challenge              | Solution/Note                                                                        |
| ---------------------- | ------------------------------------------------------------------------------------ |
| Small Dataset       | Hand-built datasets, but still prone to overfitting. Acceptable for the portfolio.       |
| Emotion Accuracy    | Emotion model doesn’t generalize well – requires a larger, diverse dataset.            |
| Similar Faces | Custom cosine similarity + threshold-based matching ensures robustness.              |
| Webcam Integration  | Real-time video + prediction flow in Streamlit achieved with `st.empty()` and `cv2`. |
| Spoof Attacks       | Added spoof head and verified using sigmoid-based classification logic.              |

---

## Installation & Usage
- Requirements
```bash
pip install -r requirements.txt
```

```bash
torch
transformers
torchvision
streamlit
opencv-python
pillow
numpy
scikit-learn
pandas
```

- Running the App
```bash
streamlit run app.py
```
> Make sure your webcam is accessible, and adjust the camera index (in app.py) to your camera. Use the Streamlit UI to interact with the system.

---

## Folder Structure
```bash
project-root/
│
├── notebooks/
│   ├── saved_embeddings/
│     └── face_embeddings.npy
│     └── face_names.npy
│   ├── saved_model/
│     └── triple_head_vit.pth
│   ├── train_face_head.ipynb
│   ├── train_emotion_head.ipynb
│   ├── train_spoof_head.ipynb
│   ├── train_dual_head_model.ipynb
│   ├── train_triple_head.ipynb
│
│
├── registered_faces/
│   └── PersonName/*.jpg
│
│
├── attendance_log.csv
│
├── face_model.py
├── emotion_model.py
├── app.py
└── README.md
```

---

## ⭐️ Give it a Star

If you found this repo helpful or interesting, please consider giving it a ⭐️. It motivates me to keep learning and sharing!

---
