"""
Smart Plate - Khana Dataset (8 Classes) Classifier Training
Designed to be run in a Google Colab Notebook.

# SETUP INSTRUCTIONS FOR GOOGLE COLAB:
# 1. Create a folder in your Google Drive named: SmartPlate_Khana
# 2. Upload the following files from your local machine to that Drive folder:
#    - dataset.tar.gz
#    - train.json
#    - val.json
#    - test.json
# 3. Create a new Google Colab notebook, set Hardware Accelerator to T4 GPU.
# 4. Copy and paste this entire script into a cell and run it.
"""

import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
import subprocess

# ==========================================
# 1. Colab Setup & File Extraction
# ==========================================
try:
    from google.colab import drive
    print("Mounting Google Drive...")
    drive.mount('/content/drive')
    
    DRIVE_PATH = "/content/drive/MyDrive/SmartPlate_Khana"
    EXTRACT_PATH = "/content/khana_data"
    
    if not os.path.exists(EXTRACT_PATH):
        print(f"Extracting dataset.tar.gz from {DRIVE_PATH} to local Colab storage for faster reading...")
        os.makedirs(EXTRACT_PATH, exist_ok=True)
        subprocess.run(["tar", "-xzf", f"{DRIVE_PATH}/dataset.tar.gz", "-C", EXTRACT_PATH], check=True)
        print("Extraction complete.")
except ImportError:
    print("Not running in Google Colab. Assuming local paths.")
    DRIVE_PATH = r"C:\Projects\SmartPlate\ml\datasets\processed\Khana_Splits"
    EXTRACT_PATH = r"C:\Projects\SmartPlate\ml\datasets\processed\khana"

# ==========================================
# 2. Dataset Definition
# ==========================================
class KhanaDataset(Dataset):
    def __init__(self, manifest_file, image_dir, transform=None):
        with open(manifest_file, 'r') as f:
            self.manifest = json.load(f)
        self.image_dir = image_dir
        self.transform = transform
        
        # We only have 8 populated classes. We map them to indices 0-7.
        self.classes = sorted(list(set(item['class'] for item in self.manifest)))
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
        
    def __len__(self):
        return len(self.manifest)
        
    def __getitem__(self, idx):
        item = self.manifest[idx]
        img_path = os.path.join(self.image_dir, item['path'].replace("\\", "/"))
        
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception:
            # Fallback to a black image if file is heavily corrupted during extract
            image = Image.new('RGB', (224, 224))
            
        label = self.class_to_idx[item['class']]
        
        if self.transform:
            image = self.transform(image)
            
        return image, label

# Transforms (Standard ImageNet prep for MobileNetV3)
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

val_test_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Create datasets and dataloaders
print("Loading datasets...")
train_dataset = KhanaDataset(os.path.join(DRIVE_PATH, "train.json"), EXTRACT_PATH, transform=train_transform)
val_dataset = KhanaDataset(os.path.join(DRIVE_PATH, "val.json"), EXTRACT_PATH, transform=val_test_transform)
test_dataset = KhanaDataset(os.path.join(DRIVE_PATH, "test.json"), EXTRACT_PATH, transform=val_test_transform)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=2)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=2)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=2)

class_names = train_dataset.classes
num_classes = len(class_names)
print(f"Classes ({num_classes}): {class_names}")

# ==========================================
# 3. Model Initialization (MobileNetV3 Small)
# ==========================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Lightweight model suitable for Colab/Mobile deployment
model = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)

# Replace final classifier layer
in_features = model.classifier[3].in_features
model.classifier[3] = nn.Linear(in_features, num_classes)

model = model.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

# ==========================================
# 4. Training Loop
# ==========================================
epochs = 10
train_losses, val_losses = [], []
train_accs, val_accs = [], []

best_val_acc = 0.0
best_model_path = os.path.join(DRIVE_PATH, "khana_mobilenetv3_best.pt")

print("Starting training...")
for epoch in range(epochs):
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
    train_loss = running_loss / total
    train_acc = correct / total
    train_losses.append(train_loss)
    train_accs.append(train_acc)
    
    # Validation Phase
    model.eval()
    val_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            val_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
    val_loss = val_loss / total
    val_acc = correct / total
    val_losses.append(val_loss)
    val_accs.append(val_acc)
    
    scheduler.step()
    
    print(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}")
    
    # Save best model
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        print(f"  -> New best validation accuracy! Saving model to {best_model_path}")
        try:
            torch.save(model.state_dict(), best_model_path)
        except Exception as e:
            print(f"  -> Error saving to Drive: {e}")

# ==========================================
# 5. Evaluation & Plots
# ==========================================
print("\nLoading best model for testing...")
try:
    model.load_state_dict(torch.load(best_model_path, weights_only=True))
except:
    pass
model.eval()

y_true, y_pred = [], []
print("Evaluating on test set...")
with torch.no_grad():
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        _, predicted = outputs.max(1)
        
        y_true.extend(labels.cpu().numpy())
        y_pred.extend(predicted.cpu().numpy())

# Classification Report
print("\n--- Test Set Evaluation ---")
report = classification_report(y_true, y_pred, target_names=class_names, digits=4)
print(report)

# Confusion Matrix Plot
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix on Test Set')
plt.tight_layout()
plt.savefig(os.path.join(DRIVE_PATH, "confusion_matrix.png"))
plt.show()

# Training Curves Plot
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Val Loss')
plt.legend()
plt.title('Loss Curves')

plt.subplot(1, 2, 2)
plt.plot(train_accs, label='Train Acc')
plt.plot(val_accs, label='Val Acc')
plt.legend()
plt.title('Accuracy Curves')

plt.tight_layout()
plt.savefig(os.path.join(DRIVE_PATH, "training_curves.png"))
plt.show()

print(f"\nTraining complete. Model and plots saved to {DRIVE_PATH}")
