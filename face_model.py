import torch
import torch.nn as nn
from transformers import ViTModel

class DualHeadViT(nn.Module):
    def __init__(self, vit, face_classes, emotion_classes):
        super().__init__()
        self.vit = vit
        self.face_head = nn.Linear(vit.config.hidden_size, face_classes)
        self.emotion_head = nn.Linear(vit.config.hidden_size, emotion_classes)

    def forward(self, x):
        features = self.vit(pixel_values=x).last_hidden_state[:, 0]
        face_out = self.face_head(features)
        emotion_out = self.emotion_head(features)
        return face_out, emotion_out


def load_model(model_path, device):
    checkpoint = torch.load(model_path, map_location=device)
    vit = ViTModel.from_pretrained("google/vit-base-patch16-224")
    face_classes = len(checkpoint['face_classes'])
    emotion_classes = len(checkpoint['emotion_classes'])

    model = DualHeadViT(vit, face_classes, emotion_classes)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    return model, checkpoint['face_classes'], checkpoint['emotion_classes']
