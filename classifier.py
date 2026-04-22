import torch
import torchvision.transforms as transforms
from torchvision import models
import torch.nn as nn
import cv2

# =========================
# LOAD MODEL
# =========================

MODEL_PATH = "../models/passenger_classifier.pt"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# safer load
checkpoint = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)

class_names = checkpoint["class_names"]

# updated model init (no deprecated warning)
model = models.resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, len(class_names))

model.load_state_dict(checkpoint["model_state_dict"])
model = model.to(DEVICE)
model.eval()

# =========================
# TRANSFORM
# =========================

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
])

# =========================
# CLASSIFICATION FUNCTION
# =========================

def classify_passenger(img):

    if img is None or img.size == 0:
        return "Unknown"

    try:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = transform(img).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            outputs = model(img)
            _, pred = torch.max(outputs, 1)

        return class_names[pred.item()]

    except:
        return "Unknown"