# ==================== INSTALLS ====================
# !pip install torch torchvision torchaudio
# !pip install seaborn tqdm joblib scikit-learn

import os
import random
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset, random_split
from torchvision import datasets, transforms, models
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split

# ==================== CONFIG ====================
DATA_DIR = "/content/drive/MyDrive/Tomato disease detection/Data"
OUTPUT_MODEL = "/content/drive/MyDrive/tomato_resnet18.pth"

RANDOM_SEED = 42
BATCH_SIZE = 32
EPOCHS = 20
LR = 1e-4
IMG_SIZE = 224
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("✅ Device:", DEVICE)

# Reproducibility
torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)
if DEVICE.type == "cuda":
    torch.cuda.manual_seed_all(RANDOM_SEED)

# ==================== TRANSFORMS ====================
train_tfms = transforms.Compose([
    transforms.RandomResizedCrop(IMG_SIZE, scale=(0.8, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(25),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

val_tfms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# ==================== DATASETS ====================
full_dataset = datasets.ImageFolder(DATA_DIR, transform=train_tfms)
class_names = full_dataset.classes
NUM_CLASSES = len(class_names)
print("🔎 Classes:", class_names)

# Stratified Split
targets = full_dataset.targets
indices = np.arange(len(full_dataset))

train_idx, temp_idx, _, temp_y = train_test_split(
    indices, targets, stratify=targets, test_size=0.3, random_state=RANDOM_SEED
)
val_idx, test_idx, _, _ = train_test_split(
    temp_idx, temp_y, stratify=temp_y, test_size=0.5, random_state=RANDOM_SEED
)

train_ds = Subset(full_dataset, train_idx)
val_ds   = Subset(datasets.ImageFolder(DATA_DIR, transform=val_tfms), val_idx)
test_ds  = Subset(datasets.ImageFolder(DATA_DIR, transform=val_tfms), test_idx)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=False)
val_loader   = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=False)
test_loader  = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=False)

# ==================== MODEL (Transfer Learning + Fine-tuning) ====================
from torchvision.models import ResNet18_Weights
model = models.resnet18(weights=ResNet18_Weights.DEFAULT)

# Freeze everything first
for param in model.parameters():
    param.requires_grad = False

# Unfreeze last block for fine-tuning
for param in model.layer4.parameters():
    param.requires_grad = True

# Replace classifier
model.fc = nn.Sequential(
    nn.Linear(model.fc.in_features, 256),
    nn.ReLU(),
    nn.Dropout(0.4),
    nn.Linear(256, NUM_CLASSES)
)
model = model.to(DEVICE)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=LR)

# ==================== EVALUATION FUNCTION ====================
def evaluate(loader):
    model.eval()
    correct, total, running_loss = 0, 0, 0
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            outputs = model(xb)
            loss = criterion(outputs, yb)
            _, preds = torch.max(outputs, 1)
            correct += (preds == yb).sum().item()
            total += yb.size(0)
            running_loss += loss.item() * xb.size(0)
    return correct/total, running_loss/total

# ==================== TRAINING LOOP ====================
train_acc_hist, val_acc_hist = [], []
train_loss_hist, val_loss_hist = [], []
best_acc = 0.0

for epoch in range(1, EPOCHS+1):
    model.train()
    correct_train, total_train, running_loss = 0, 0, 0

    for xb, yb in train_loader:
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(xb)
        loss = criterion(outputs, yb)
        loss.backward()
        optimizer.step()

        _, preds = torch.max(outputs, 1)
        correct_train += (preds == yb).sum().item()
        total_train += yb.size(0)
        running_loss += loss.item() * xb.size(0)

    # Epoch metrics
    train_acc = correct_train / total_train
    train_loss = running_loss / total_train
    val_acc, val_loss = evaluate(val_loader)

    # Store history
    train_acc_hist.append(train_acc)
    val_acc_hist.append(val_acc)
    train_loss_hist.append(train_loss)
    val_loss_hist.append(val_loss)

    # Print summary
    print(f"Epoch {epoch:02d}/{EPOCHS} | "
          f"Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}, "
          f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

    # Save best model
    if val_acc > best_acc:
        best_acc = val_acc
        torch.save(model.state_dict(), OUTPUT_MODEL)

print("🏁 Training complete. Best Val Acc:", best_acc)

# ==================== TEST EVALUATION ====================
model.load_state_dict(torch.load(OUTPUT_MODEL))
model.eval()

all_preds, all_labels = [], []
with torch.no_grad():
    for xb, yb in test_loader:
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
        outputs = model(xb)
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(yb.cpu().numpy())

# Confusion Matrix
cm = confusion_matrix(all_labels, all_preds)
plt.figure(figsize=(12, 10))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=class_names,
            yticklabels=class_names)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

# Classification Report
print("📊 Classification Report:\n",
      classification_report(all_labels, all_preds, target_names=class_names, zero_division=0))

# ==================== PLOTS ====================
plt.figure(figsize=(10,5))
plt.plot(train_acc_hist, label="Train Acc")
plt.plot(val_acc_hist, label="Val Acc")
plt.legend()
plt.title("Accuracy over Epochs")
plt.show()

plt.figure(figsize=(10,5))
plt.plot(train_loss_hist, label="Train Loss")
plt.plot(val_loss_hist, label="Val Loss")
plt.legend()
plt.title("Loss over Epochs")
plt.show()
