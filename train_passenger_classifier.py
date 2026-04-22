import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, random_split
import os

# =========================
# CONFIG
# =========================

DATA_DIR = "../datasets/passenger_dataset/train"
MODEL_SAVE_PATH = "../models/passenger_classifier.pt"

BATCH_SIZE = 32
EPOCHS = 20
LR = 0.0003
IMG_SIZE = 128

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print("Using device:", DEVICE)

# =========================
# TRANSFORMS
# =========================

train_transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.3, contrast=0.3),
    transforms.GaussianBlur(3),
    transforms.RandomAffine(0, translate=(0.1, 0.1)),
    transforms.ToTensor(),
])

val_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
])

# =========================
# DATASET
# =========================

full_dataset = datasets.ImageFolder(DATA_DIR, transform=train_transform)

train_size = int(0.8 * len(full_dataset))
val_size = len(full_dataset) - train_size

train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

val_dataset.dataset.transform = val_transform

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

class_names = full_dataset.classes
print("Classes:", class_names)

# =========================
# MODEL (ResNet18)
# =========================

model = models.resnet18(pretrained=True)

model.fc = nn.Linear(model.fc.in_features, len(class_names))

model = model.to(DEVICE)

# =========================
# LOSS & OPTIMIZER
# =========================

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LR)

# =========================
# TRAIN LOOP
# =========================

for epoch in range(EPOCHS):

    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for images, labels in train_loader:

        images, labels = images.to(DEVICE), labels.to(DEVICE)

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    train_acc = 100 * correct / total

    # VALIDATION
    model.eval()
    val_correct = 0
    val_total = 0

    with torch.no_grad():
        for images, labels in val_loader:

            images, labels = images.to(DEVICE), labels.to(DEVICE)

            outputs = model(images)

            _, preds = torch.max(outputs, 1)

            val_correct += (preds == labels).sum().item()
            val_total += labels.size(0)

    val_acc = 100 * val_correct / val_total

    print(f"Epoch {epoch+1}/{EPOCHS}")
    print(f"Train Loss: {total_loss:.4f} | Train Acc: {train_acc:.2f}%")
    print(f"Val Acc: {val_acc:.2f}%")
    print("-"*40)

# =========================
# SAVE MODEL
# =========================

torch.save({
    "model_state_dict": model.state_dict(),
    "class_names": class_names
}, MODEL_SAVE_PATH)

print("Model saved to:", MODEL_SAVE_PATH)