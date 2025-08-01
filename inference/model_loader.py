# Model loading logic

import torch
import torch.nn as nn
from torchvision import datasets, transforms
from transformers import ViTModel

DEVICE = 'cuda'
MODEL_PATH = "./notebooks/saved_model/triple_head_vit.pth"
FACE_DIR = "./registered_faces"
EMOTION_DIR = "./emotions_data"

# Transforms
base_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

emotion_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor()
])

face_dataset = datasets.ImageFolder(FACE_DIR, transform=base_transform)
emotion_dataset = datasets.ImageFolder(EMOTION_DIR, transform=emotion_transform)

def load_triple_head_model(MODEL_PATH, DEVICE):
    # Model
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
    
    vit = ViTModel.from_pretrained("google/vit-base-patch16-224")
    model = TripleHeadViT(vit, len(face_dataset.classes), len(emotion_dataset.classes)).to(DEVICE)
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model

# model = load_triple_head_model(MODEL_PATH, DEVICE)
# print(model)
