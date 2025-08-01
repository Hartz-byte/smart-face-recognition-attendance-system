import torch.nn.functional as F

# Emotion classes
EMOTION_CLASSES = [
    "neutral",
    "happy",
    "sad",
    "angry",
    "surprised",
    "fearful",
    "disgusted"
]

def decode_emotion(emotion_logits):
    """
    Given logits from emotion head, return (emotion label, confidence score)
    """
    # Convert logits to probability using softmax
    probs = F.softmax(emotion_logits, dim=0)
    max_index = probs.argmax().item()
    emotion = EMOTION_CLASSES[max_index]
    score = probs[max_index].item()
    return emotion, score
