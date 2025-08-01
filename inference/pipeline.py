# Unified inference logic

from torchvision import transforms
import torch
# from PIL import Image
# from model_loader import load_triple_head_model

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

def run_inference(model, image, device):
    tensor = transform(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        face_logits, emotion_logits, spoof_logits = model(tensor)
        features = model.vit(pixel_values=tensor).last_hidden_state[:, 0]
        spoof_prob = torch.sigmoid(spoof_logits).item()
        
        return {
            "features": features[0].cpu().numpy(),
            "emotion_logits": emotion_logits[0].cpu(),
            "spoof_prob": spoof_prob
        }

# model = load_triple_head_model("../notebooks/saved_model/triple_head_vit.pth", "cuda")
# img = Image.open("../dp.jpeg")
# output = run_inference(model, img, "cuda")
# print(output)
