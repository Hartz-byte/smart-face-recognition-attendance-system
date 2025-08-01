from fastapi import FastAPI, File, UploadFile, HTTPException, Form, BackgroundTasks
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

from PIL import Image
import io
import numpy as np
import torch
from sklearn.metrics.pairwise import cosine_similarity
from typing import List
import os
import logging

from inference.model_loader import load_triple_head_model
from inference.pipeline import run_inference
from inference.utils import load_embeddings, mark_attendance
from inference.emotion_model import decode_emotion

# Config
MODEL_PATH = "./notebooks/saved_model/triple_head_vit.pth"
EMBEDDINGS_FILE = "./notebooks/saved_embeddings/face_embeddings.npy"
NAMES_FILE = "./notebooks/saved_embeddings/face_names.npy"
ATTENDANCE_CSV = "./attendance_log.csv"
SIMILARITY_THRESHOLD = 0.60
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load Model and Embeddings once at startup
model = load_triple_head_model(MODEL_PATH, DEVICE)
embeddings, names = load_embeddings(EMBEDDINGS_FILE, NAMES_FILE)

# Initialize FastAPI
app = FastAPI(title="Smart Attendance API")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Embedding Update Logic
def recompute_embeddings(model, device, registered_dir, embeddings_file, names_file):
    try:
        from torch.utils.data import DataLoader
        from torchvision import datasets, transforms

        dataset = datasets.ImageFolder(
            root=registered_dir,
            transform=transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor()
            ])
        )
        loader = DataLoader(dataset, batch_size=4)

        all_embeddings, all_names = [], []
        with torch.no_grad():
            for imgs, labels in loader:
                imgs = imgs.to(device)
                features = model.vit(pixel_values=imgs).last_hidden_state[:, 0]
                all_embeddings.append(features.cpu().numpy())
                all_names += [dataset.classes[label] for label in labels]

        np.save(embeddings_file, np.vstack(all_embeddings))
        np.save(names_file, np.array(all_names))

        # Reload globals
        global embeddings, names
        embeddings, names = load_embeddings(embeddings_file, names_file)
        
    except Exception as e:
        logging.exception(f"Error updating embeddings: {e}")


@app.get("/")
async def root():
    return {"message": "Smart Attendance API is running"}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    Accepts an uploaded image file,
    returns predicted name, emotion, spoof score.
    """

    # Check file type
    if file.content_type not in ["image/jpeg", "image/png"]:
        raise HTTPException(status_code=400, detail="Invalid image format")

    # Read image bytes
    img_bytes = await file.read()
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")

    # Run inference
    output = run_inference(model, img, DEVICE)
    emb = output["features"]
    emotion_logits = output["emotion_logits"]
    spoof_score = output["spoof_prob"]

    result = {}

    # Spoof detection threshold
    if spoof_score > 0.5:
        result["spoof"] = True
        result["spoof_score"] = spoof_score
        result["message"] = "Spoof detected!"
        return JSONResponse(content=result, status_code=200)

    # Face recognition (similarity check)
    sims = cosine_similarity([emb], embeddings)[0]
    max_idx = np.argmax(sims)
    max_sim = sims[max_idx]

    if max_sim > SIMILARITY_THRESHOLD:
        name = names[max_idx]
        mark_attendance(name, ATTENDANCE_CSV)
    else:
        name = "Unknown"

    # Decode emotion
    emotion, emotion_score = decode_emotion(emotion_logits)

    # Compose response
    result.update({
        "spoof": False,
        "spoof_score": spoof_score,
        "name": name,
        "similarity": float(max_sim),
        "emotion": emotion,
        "emotion_score": float(emotion_score)
    })

    return result


@app.get("/attendance")
async def get_attendance():
    """
    Returns the attendance log as JSON.
    """
    if not os.path.exists(ATTENDANCE_CSV):
        return {"attendance": [], "message": "No attendance log available."}

    import pandas as pd
    df = pd.read_csv(ATTENDANCE_CSV)
    attendance_list = df.to_dict(orient="records")
    return {"attendance": attendance_list}


@app.post("/update-embeddings")
async def update_embeddings(background_tasks: BackgroundTasks):
    """
    Triggers background task to recompute embeddings. Returns immediately.
    """
    REGISTERED_DIR = "./registered_faces"

    # Schedule the task to run in the background
    background_tasks.add_task(
        recompute_embeddings,
        model, DEVICE, REGISTERED_DIR, EMBEDDINGS_FILE, NAMES_FILE
    )
    return {"message": "Embeddings update task started. You can continue using the service."}

@app.post("/add-person")
async def add_person(
    name: str = Form(...), 
    files: List[UploadFile] = File(...)
):
    """
    Registers a new person by saving uploaded face images 
    into a folder named after the person.
    Expects: name (form field) and one or more image files as files[].
    """
    REGISTERED_DIR = "./registered_faces"
    save_dir = os.path.join(REGISTERED_DIR, name.strip())
    os.makedirs(save_dir, exist_ok=True)

    count = 0
    for idx, file in enumerate(files):
        if file.content_type not in ["image/jpeg", "image/png"]:
            continue
        img_bytes = await file.read()
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        img.save(os.path.join(save_dir, f"{idx}.jpg"))
        count += 1

    if count == 0:
        return JSONResponse(status_code=400, content={"message": "No valid images uploaded."})

    return {
        "message": f"Successfully saved {count} images for '{name}'. Please call /update-embeddings to reflect changes."
    }