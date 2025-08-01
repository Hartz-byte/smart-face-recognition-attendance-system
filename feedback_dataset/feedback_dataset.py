import json
import shutil
import os

feedback_log = "feedback_frames/feedback_log.json"
face_out_dir = "feedback_frames/self_training/face/"
emotion_out_dir = "feedback_frames/self_training/emotion/"
os.makedirs(face_out_dir, exist_ok=True)
os.makedirs(emotion_out_dir, exist_ok=True)

with open(feedback_log, "r") as f:
    feedbacks = json.load(f)

for entry in feedbacks:
    image_path = entry["image"]
    # Face dataset (skip unknowns for now if you wish)
    if entry["corrected_name"].lower() != "unknown":
        person_folder = os.path.join(face_out_dir, entry["corrected_name"])
        os.makedirs(person_folder, exist_ok=True)
        shutil.copy(image_path, person_folder)
    # Emotion dataset
    emotion_folder = os.path.join(emotion_out_dir, entry["corrected_emotion"])
    os.makedirs(emotion_folder, exist_ok=True)
    shutil.copy(image_path, emotion_folder)
