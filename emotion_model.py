EMOTION_LABELS = {
    0: "Angry",
    1: "Disgust",
    2: "Fear",
    3: "Happy",
    4: "Sad",
    5: "Surprise",
    6: "Neutral"
}

def decode_emotion(emotion_logits):
    import torch.nn.functional as F
    probs = F.softmax(emotion_logits, dim=0)
    idx = probs.argmax().item()
    return EMOTION_LABELS.get(idx, "Unknown"), probs[idx].item()
