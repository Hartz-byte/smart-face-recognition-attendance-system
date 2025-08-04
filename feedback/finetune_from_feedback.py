# import os
# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torchvision import transforms
# from torch.utils.data import DataLoader, Dataset
# from transformers import ViTModel
# from tqdm import tqdm
# import json
# from PIL import Image

# # CONFIG
# FACE_FEEDBACK_DIR = "./feedback_frames/self_training/face"
# EMOTION_FEEDBACK_DIR = "./feedback_frames/self_training/emotion"
# MODEL_PATH = "./notebooks/saved_model/triple_head_vit.pth"
# BATCH_SIZE = 8
# EPOCHS = 3
# LR = 5e-5   # Lower rate to avoid catastrophic forgetting
# DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# # LOAD ORIGINAL CLASS LISTS
# with open('./notebooks/saved_model/face_classes.json', 'r') as f:
#     orig_face_classes = json.load(f)
# with open('./notebooks/saved_model/emotion_classes.json', 'r') as f:
#     orig_emotion_classes = json.load(f)

# # DATA TRANSFORMS
# base_transform = transforms.Compose([
#     transforms.Resize((224, 224)),
#     transforms.ToTensor()
# ])

# # CUSTOM DATASET FOR STRICT CLASS MATCHING
# class FeedbackDataset(Dataset):
#     def __init__(self, base_dir, class_names, transform):
#         self.transform = transform
#         self.samples = []
#         for cls in class_names:
#             cls_dir = os.path.join(base_dir, cls)
#             if os.path.isdir(cls_dir):
#                 for img_name in os.listdir(cls_dir):
#                     img_path = os.path.join(cls_dir, img_name)
#                     if img_path.lower().endswith(('.jpg', '.jpeg', '.png')):
#                         self.samples.append((img_path, class_names.index(cls)))
#     def __len__(self):
#         return len(self.samples)
#     def __getitem__(self, idx):
#         path, label = self.samples[idx]
#         img = Image.open(path).convert('RGB')
#         return self.transform(img), label

# face_feedback_dataset = FeedbackDataset(FACE_FEEDBACK_DIR, orig_face_classes, base_transform)
# emotion_feedback_dataset = FeedbackDataset(EMOTION_FEEDBACK_DIR, orig_emotion_classes, base_transform)

# face_loader = DataLoader(face_feedback_dataset, batch_size=BATCH_SIZE, shuffle=True)
# emotion_loader = DataLoader(emotion_feedback_dataset, batch_size=BATCH_SIZE, shuffle=True)

# # MODEL DEFINITION
# class TripleHeadViT(nn.Module):
#     def __init__(self, vit, face_classes, emotion_classes):
#         super().__init__()
#         self.vit = vit
#         self.dropout = nn.Dropout(0.3)
#         self.face_head = nn.Linear(vit.config.hidden_size, face_classes)
#         self.emotion_head = nn.Linear(vit.config.hidden_size, emotion_classes)
#         self.spoof_head = nn.Linear(vit.config.hidden_size, 1)
#     def forward(self, x):
#         features = self.vit(pixel_values=x).last_hidden_state[:, 0]
#         features = self.dropout(features)
#         return self.face_head(features), self.emotion_head(features), self.spoof_head(features)

# # LOAD PREVIOUS MODEL
# ckpt = torch.load(MODEL_PATH, map_location=DEVICE)
# vit = ViTModel.from_pretrained("google/vit-base-patch16-224")
# for name, param in vit.named_parameters():
#     if "encoder.layer.11" not in name and "encoder.layer.10" not in name:
#         param.requires_grad = False

# model = TripleHeadViT(vit, len(orig_face_classes), len(orig_emotion_classes)).to(DEVICE)
# model.load_state_dict(ckpt["model_state_dict"], strict=True)
# model.train()

# # LOSSES & OPTIMIZER
# face_criterion = nn.CrossEntropyLoss()
# emotion_criterion = nn.CrossEntropyLoss()
# optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=LR)

# # TRAINING LOOP
# for epoch in range(EPOCHS):
#     print(f"\nAdaptive Fine-Tune Epoch {epoch+1}/{EPOCHS}")
#     face_iter = iter(face_loader)
#     emotion_iter = iter(emotion_loader)
#     steps = min(len(face_iter), len(emotion_iter))
#     total_loss = 0

#     for _ in tqdm(range(steps), desc=f"Epoch {epoch+1}"):
#         x_face, y_face = next(face_iter)
#         x_emotion, y_emotion = next(emotion_iter)
#         x_face, y_face = x_face.to(DEVICE), y_face.to(DEVICE)
#         x_emotion, y_emotion = x_emotion.to(DEVICE), y_emotion.to(DEVICE)

#         x = torch.cat([x_face, x_emotion], dim=0)
#         face_logits, emotion_logits, _ = model(x)
#         face_loss = face_criterion(face_logits[:len(y_face)], y_face)
#         emotion_loss = emotion_criterion(emotion_logits[len(y_face):], y_emotion)
#         loss = face_loss + emotion_loss

#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
#         total_loss += loss.item()

#     print(f"  [Fine-Tune] Avg Loss: {total_loss/steps:.4f}")

# # SAVE UPDATED MODEL
# torch.save({
#     "model_state_dict": model.state_dict(),
#     "face_classes": orig_face_classes,
#     "emotion_classes": orig_emotion_classes
# }, MODEL_PATH)
# print(f"\n✔️ Model weights updated with new feedback samples and saved to {MODEL_PATH}")


import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, ConcatDataset
from transformers import ViTModel
from tqdm import tqdm
import json
from PIL import Image

# CONFIG
FACE_DIR = "./registered_faces"
EMOTION_DIR = "./emotions_data"
FEEDBACK_FACE_DIR = "./feedback_frames/self_training/face"
FEEDBACK_EMOTION_DIR = "./feedback_frames/self_training/emotion"
MODEL_PATH = "./notebooks/saved_model/triple_head_vit.pth"
BATCH_SIZE = 8
EPOCHS = 3
LR = 5e-5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load original class lists
with open('./notebooks/saved_model/face_classes.json', 'r') as f:
    orig_face_classes = json.load(f)
with open('./notebooks/saved_model/emotion_classes.json', 'r') as f:
    orig_emotion_classes = json.load(f)

# Transforms
original_face_transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor()
])
original_emotion_transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor()
])
feedback_transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor()
])

class FeedbackDataset(torch.utils.data.Dataset):
    def __init__(self, base_dir, class_names, transform):
        self.transform = transform
        self.samples = []
        for cls in class_names:
            cls_dir = os.path.join(base_dir, cls)
            if os.path.isdir(cls_dir):
                for img_name in os.listdir(cls_dir):
                    if img_name.lower().endswith((".jpg", ".jpeg", ".png")):
                        self.samples.append((os.path.join(cls_dir, img_name), class_names.index(cls)))
    def __len__(self):
        return len(self.samples)
    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        return self.transform(img), label

# Load datasets
original_face_dataset = datasets.ImageFolder(FACE_DIR, original_face_transform)
original_emotion_dataset = datasets.ImageFolder(EMOTION_DIR, original_emotion_transform)
feedback_face_dataset = FeedbackDataset(FEEDBACK_FACE_DIR, orig_face_classes, feedback_transform)
feedback_emotion_dataset = FeedbackDataset(FEEDBACK_EMOTION_DIR, orig_emotion_classes, feedback_transform)

# Combine datasets
combined_face_dataset = ConcatDataset([original_face_dataset, feedback_face_dataset])
combined_emotion_dataset = ConcatDataset([original_emotion_dataset, feedback_emotion_dataset])

# DataLoaders
face_loader = DataLoader(combined_face_dataset, batch_size=BATCH_SIZE, shuffle=True)
emotion_loader = DataLoader(combined_emotion_dataset, batch_size=BATCH_SIZE, shuffle=True)

# MODEL DEFINITION
class TripleHeadViT(nn.Module):
    def __init__(self, vit, face_classes, emotion_classes):
        super().__init__()
        self.vit = vit
        self.dropout = nn.Dropout(0.3)
        self.face_head = nn.Linear(vit.config.hidden_size, face_classes)
        self.emotion_head = nn.Linear(vit.config.hidden_size, emotion_classes)
        self.spoof_head = nn.Linear(vit.config.hidden_size, 1)
    def forward(self, x):
        features = self.vit(pixel_values=x).last_hidden_state[:, 0]
        features = self.dropout(features)
        return self.face_head(features), self.emotion_head(features), self.spoof_head(features)

# Load Model
ckpt = torch.load(MODEL_PATH, map_location=DEVICE)
vit = ViTModel.from_pretrained("google/vit-base-patch16-224")
for name, param in vit.named_parameters():
    if "encoder.layer.11" not in name and "encoder.layer.10" not in name:
        param.requires_grad = False

model = TripleHeadViT(vit, len(orig_face_classes), len(orig_emotion_classes)).to(DEVICE)
model.load_state_dict(ckpt["model_state_dict"], strict=True)
model.train()

# Losses and Optimizer
criterion_face = nn.CrossEntropyLoss()
criterion_emotion = nn.CrossEntropyLoss()
optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=LR)

# TRAINING LOOP
for epoch in range(EPOCHS):
    print(f"Epoch {epoch+1}/{EPOCHS}")
    face_correct = 0
    emotion_correct = 0
    face_total = 0
    emotion_total = 0
    total_loss = 0
    face_iter = iter(face_loader)
    emotion_iter = iter(emotion_loader)
    steps = min(len(face_iter), len(emotion_iter))
    for _ in tqdm(range(steps)):
        x_face, y_face = next(face_iter)
        x_emotion, y_emotion = next(emotion_iter)
        x_face, y_face = x_face.to(DEVICE), y_face.to(DEVICE)
        x_emotion, y_emotion = x_emotion.to(DEVICE), y_emotion.to(DEVICE)
        x = torch.cat([x_face, x_emotion], dim=0)
        optimizer.zero_grad()
        face_logits, emotion_logits, _ = model(x)
        loss_face = criterion_face(face_logits[:len(y_face)], y_face)
        loss_emotion = criterion_emotion(emotion_logits[len(y_face):], y_emotion)
        loss = loss_face + loss_emotion
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        face_pred = face_logits[:len(y_face)].argmax(dim=1)
        emotion_pred = emotion_logits[len(y_face):].argmax(dim=1)
        face_correct += (face_pred == y_face).sum().item()
        emotion_correct += (emotion_pred == y_emotion).sum().item()
        face_total += y_face.size(0)
        emotion_total += y_emotion.size(0)
    print(f"Loss: {total_loss / steps:.4f}")
    print(f"Face Accuracy: {face_correct / face_total * 100:.2f}%")
    print(f"Emotion Accuracy: {emotion_correct / emotion_total * 100:.2f}%")

# SAVE MODEL
torch.save({
    "model_state_dict": model.state_dict(),
    "face_classes": orig_face_classes,
    "emotion_classes": orig_emotion_classes
}, MODEL_PATH)
print(f"Model saved to: {MODEL_PATH}")
