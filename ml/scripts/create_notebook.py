import json
import os

cells = []

def add_markdown(text):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": [line + "\n" for line in text.split("\n")]})

def add_code(text):
    cells.append({"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": [line + "\n" for line in text.split("\n")]})

add_markdown("""# Smart Plate - Robust Food Classification Training
This notebook trains a MobileNetV3-Large model on a combined dataset of 40,000+ Indian food images.
**Features:**
- Downloads from HuggingFace and Kaggle.
- Extracts local YOLO/Classification datasets from Google Drive.
- YOLO Crop parsing (crops bounding boxes into classification images).
- Explicit class mapping and filtering.
- Deduplication (MD5 hashing) to prevent train/val leakage.
- Stratified 80/10/10 split removing micro-classes.
- 1-Epoch Smoke Test (with state restoration).
- Resumable Training from checkpoints.
- Full Evaluation (F1, Confusion Matrix, Curves).""")

add_code("""# 1. Setup & Authentication
!pip install -q datasets opendatasets ultralytics scikit-learn pyyaml

import os
import hashlib
import shutil
import glob
import json
import yaml
import copy
from collections import Counter
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader, Subset
from datasets import load_dataset
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

from google.colab import drive
drive.mount('/content/drive')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')
""")

add_code("""# 2. Dataset Acquisition
RAW_DIR = '/content/raw_data'
COMBINED_DIR = '/content/combined_data'
DRIVE_DIR = '/content/drive/MyDrive/Smartplate'
EXP_DIR = os.path.join(DRIVE_DIR, 'experiments/experiment_classification_1')
os.makedirs(RAW_DIR, exist_ok=True)
os.makedirs(COMBINED_DIR, exist_ok=True)
os.makedirs(DRIVE_DIR, exist_ok=True)
os.makedirs(EXP_DIR, exist_ok=True)

print('--- Downloading Enhanced Indian Food Classification (HF) ---')
try:
    hf_dataset = load_dataset('SohlHealth/enhanced-indian-food-classification', split='train')
    hf_dir = os.path.join(RAW_DIR, 'hf_enhanced')
    os.makedirs(hf_dir, exist_ok=True)
    
    # Try to get class names mapping if available
    int2str = None
    if hasattr(hf_dataset.features.get('label', None), 'int2str'):
        int2str = hf_dataset.features['label'].int2str

    for idx, item in enumerate(hf_dataset):
        try:
            img = item['image'].convert('RGB')
            label_val = item.get('label', item.get('class', 'unknown'))
            
            if int2str is not None and isinstance(label_val, int):
                label = int2str(label_val)
            else:
                label = str(label_val)
                
            label = label.lower().strip().replace(' ', '_')
            cls_dir = os.path.join(hf_dir, label)
            os.makedirs(cls_dir, exist_ok=True)
            img.save(os.path.join(cls_dir, f'hf_{idx}.jpg'))
        except Exception as e:
            # Skip corrupted individual images without aborting the loop
            continue
    print('HF dataset downloaded and extracted.')
except Exception as e:
    print('Error loading HF dataset entirely:', e)

print('\\n--- Downloading Kaggle Dataset ---')
kaggle_cred_path = '/root/.kaggle/kaggle.json'
if os.path.exists('/content/drive/MyDrive/kaggle.json'):
    os.makedirs('/root/.kaggle', exist_ok=True)
    shutil.copy('/content/drive/MyDrive/kaggle.json', kaggle_cred_path)
    os.chmod(kaggle_cred_path, 0o600)
    print('Kaggle credentials found. Downloading sudarshan24/indian-food-images-top-20...')
    !kaggle datasets download -d sudarshan24/indian-food-images-top-20 --unzip -p /content/raw_data/kaggle_20 --quiet
    print('Kaggle dataset downloaded.')
else:
    print('No kaggle.json found in Drive. Skipping Kaggle dataset.')

print('\\n--- Extracting Local Drive Datasets ---')
for archive in ['indianfoodnet30.tar.gz', 'indianfoodnet30.zip', 'smartplate_combined.zip']:
    archive_path = os.path.join(DRIVE_DIR, archive)
    if os.path.exists(archive_path):
        print(f'Extracting {archive}...')
        if archive.endswith('.zip'):
            !unzip -q "{archive_path}" -d "{RAW_DIR}"
        else:
            !tar -xzf "{archive_path}" -C "{RAW_DIR}"
    else:
        print(f'{archive} not found in {DRIVE_DIR}')
""")

add_code("""# 3. YOLO Cropping & Explicit Class Mapping
def get_image_hash(img_bytes):
    hasher = hashlib.md5()
    hasher.update(img_bytes)
    return hasher.hexdigest()

CLASS_MAPPING = {
    'chapati': 'roti_naan',
    'roti': 'roti_naan',
    'naan': 'roti_naan',
    'chicken_curry': 'curry_wet',
    'palak_paneer': 'curry_wet',
    'paneer_butter_masala': 'curry_wet',
    'dal_makhani': 'dal_sambar',
    'dal': 'dal_sambar',
    'sambar': 'dal_sambar',
    'pizza': 'EXCLUDE',
    'burger': 'EXCLUDE',
    'sushi': 'EXCLUDE'
}

unique_hashes = set()
valid_images_count = 0
excluded_count = 0
duplicate_count = 0
corrupt_count = 0

def save_image_if_valid(img, raw_class):
    global valid_images_count, excluded_count, duplicate_count, corrupt_count
    mapped_class = CLASS_MAPPING.get(raw_class, raw_class)
    if mapped_class == 'EXCLUDE':
        excluded_count += 1
        return
    try:
        img = img.convert('RGB')
        img_byte_arr = img.tobytes()
        img_hash = get_image_hash(img_byte_arr)
        if img_hash in unique_hashes:
            duplicate_count += 1
            return
        unique_hashes.add(img_hash)
        
        dest_dir = os.path.join(COMBINED_DIR, mapped_class)
        os.makedirs(dest_dir, exist_ok=True)
        img.save(os.path.join(dest_dir, f"{img_hash}.jpg"))
        valid_images_count += 1
    except Exception:
        corrupt_count += 1

print('--- Processing YOLO Datasets (Cropping bounding boxes) ---')
processed_yolo_img_dirs = set()

for yaml_path in glob.glob(os.path.join(RAW_DIR, '**', 'data.yaml'), recursive=True):
    base_dir = os.path.dirname(yaml_path)
    with open(yaml_path, 'r') as f:
        data_cfg = yaml.safe_load(f)
    
    # Handle dict or list names
    raw_names = data_cfg.get('names', [])
    if isinstance(raw_names, dict):
        names = {int(k): v for k, v in raw_names.items()}
    else:
        names = {i: v for i, v in enumerate(raw_names)}
    
    # Find all 'images' directories near this data.yaml
    for img_dir in glob.glob(os.path.join(base_dir, '**', 'images'), recursive=True):
        processed_yolo_img_dirs.add(os.path.abspath(img_dir))
        lbl_dir = img_dir.replace('images', 'labels')
        if not os.path.exists(lbl_dir): continue
        
        for img_file in os.listdir(img_dir):
            if not img_file.lower().endswith(('.jpg', '.jpeg', '.png')): continue
            img_path = os.path.join(img_dir, img_file)
            lbl_path = os.path.join(lbl_dir, os.path.splitext(img_file)[0] + '.txt')
            
            try:
                with Image.open(img_path) as img:
                    w, h = img.size
                    if os.path.exists(lbl_path):
                        with open(lbl_path, 'r') as lf:
                            for line in lf:
                                parts = line.strip().split()
                                if len(parts) >= 5:
                                    cls_id = int(parts[0])
                                    cx, cy, bw, bh = map(float, parts[1:5])
                                    if cls_id in names:
                                        cls_name = names[cls_id].lower().replace(' ', '_')
                                        left = max(0, (cx - bw/2) * w)
                                        top = max(0, (cy - bh/2) * h)
                                        right = min(w, (cx + bw/2) * w)
                                        bottom = min(h, (cy + bh/2) * h)
                                        if right > left and bottom > top:
                                            cropped = img.crop((left, top, right, bottom))
                                            save_image_if_valid(cropped, cls_name)
            except Exception:
                corrupt_count += 1

print('--- Processing Standard Classification Datasets ---')
for root, dirs, files in os.walk(RAW_DIR):
    # Skip directories that are part of YOLO image processing
    if os.path.abspath(root) in processed_yolo_img_dirs:
        continue
    
    folder_name = os.path.basename(root).lower()
    if folder_name in ['images', 'labels', 'train', 'val', 'valid', 'test']:
        continue
    
    raw_class = folder_name.strip().replace(' ', '_')
    for file in files:
        if file.lower().endswith(('.png', '.jpg', '.jpeg')):
            filepath = os.path.join(root, file)
            try:
                with Image.open(filepath) as img:
                    save_image_if_valid(img, raw_class)
            except Exception:
                corrupt_count += 1

print(f"\\nProcessing Complete:")
print(f"Valid Unique Images: {valid_images_count}")
print(f"Duplicates Avoided: {duplicate_count}")
print(f"Corrupted Images Skipped: {corrupt_count}")
print(f"Excluded Classes Skipped: {excluded_count}")
""")

add_code("""# 4. PyTorch Dataset & Stratified Split
if valid_images_count == 0:
    raise ValueError("No valid images found! Please check dataset extraction.")

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

dataset = datasets.ImageFolder(root=COMBINED_DIR, transform=transform)

# Remove classes with < 10 samples to allow stratified split
class_counts = Counter([label for _, label in dataset.samples])
valid_indices = [i for i, (_, label) in enumerate(dataset.samples) if class_counts[label] >= 10]
dataset = Subset(dataset, valid_indices)

labels = [dataset.dataset.samples[i][1] for i in dataset.indices]
unique_labels = set(labels)
print(f'Remaining Usable Classes (>= 10 samples): {len(unique_labels)}')

if len(unique_labels) < 2:
    print("WARNING: Less than 2 classes survived! Disabling stratification.")
    train_idx, temp_idx = train_test_split(list(range(len(dataset))), test_size=0.2, random_state=42)
    val_idx, test_idx = train_test_split(temp_idx, test_size=0.5, random_state=42)
else:
    train_idx, temp_idx, _, temp_labels = train_test_split(
        list(range(len(dataset))), labels, test_size=0.2, stratify=labels, random_state=42
    )
    val_idx, test_idx = train_test_split(
        temp_idx, test_size=0.5, stratify=temp_labels, random_state=42
    )

train_set = Subset(dataset, train_idx)
val_set = Subset(dataset, val_idx)
test_set = Subset(dataset, test_idx)

train_loader = DataLoader(train_set, batch_size=64, shuffle=True, num_workers=2)
val_loader = DataLoader(val_set, batch_size=64, shuffle=False, num_workers=2)
test_loader = DataLoader(test_set, batch_size=64, shuffle=False, num_workers=2)

classes = dataset.dataset.classes
num_classes = len(classes)
print(f'Train: {len(train_set)}, Val: {len(val_set)}, Test: {len(test_set)}')

print("\\nPer-Class Counts:")
for cls_idx, count in class_counts.items():
    if count >= 10:
        print(f" - {classes[cls_idx]}: {count}")
""")

add_code("""# 5. Model & Resumable Checkpointing
model = models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.DEFAULT)
in_features = model.classifier[3].in_features
model.classifier[3] = nn.Linear(in_features, num_classes)
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

CHECKPOINT_PATH = os.path.join(EXP_DIR, 'mobilenetv3_large_checkpoint.pt')
BEST_MODEL_PATH = os.path.join(EXP_DIR, 'mobilenetv3_large_best.pt')
LOGS_PATH = os.path.join(EXP_DIR, 'training_logs.json')

start_epoch = 0
best_acc = 0.0
logs = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}

if os.path.exists(CHECKPOINT_PATH):
    print(f'Resuming from checkpoint: {CHECKPOINT_PATH}')
    checkpoint = torch.load(CHECKPOINT_PATH)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint['epoch'] + 1
    best_acc = checkpoint['best_acc']
    if os.path.exists(LOGS_PATH):
        with open(LOGS_PATH, 'r') as f:
            logs = json.load(f)
else:
    print('No checkpoint found. Starting fresh.')
""")

add_code("""# 6. The 1-Epoch Smoke Test (Non-Destructive)
if start_epoch == 0:
    print('\\n--- Running 1-Epoch Smoke Test ---')
    init_model_state = copy.deepcopy(model.state_dict())
    init_opt_state = copy.deepcopy(optimizer.state_dict())
    
    model.train()
    for i, (inputs, labels) in enumerate(train_loader):
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        if i == 1: break
        
    print('Smoke test successful! Restoring initial state...')
    model.load_state_dict(init_model_state)
    optimizer.load_state_dict(init_opt_state)
    print('State restored. Ready for full training.\\n')
""")

add_code("""# 7. Main Training Loop with Checkpointing
epochs = 10

for epoch in range(start_epoch, epochs):
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
        _, preds = outputs.max(1)
        total += labels.size(0)
        correct += preds.eq(labels).sum().item()
        
    train_loss = running_loss / total
    train_acc = correct / total
    
    model.eval()
    val_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item() * inputs.size(0)
            _, preds = outputs.max(1)
            total += labels.size(0)
            correct += preds.eq(labels).sum().item()
            
    val_loss = val_loss / total
    val_acc = correct / total
    
    logs['train_loss'].append(train_loss)
    logs['train_acc'].append(train_acc)
    logs['val_loss'].append(val_loss)
    logs['val_acc'].append(val_acc)
    
    scheduler.step()
    print(f'Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}')
    
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'best_acc': best_acc
    }, CHECKPOINT_PATH)
    
    with open(LOGS_PATH, 'w') as f:
        json.dump(logs, f)
        
    if val_acc > best_acc:
        best_acc = val_acc
        torch.save(model.state_dict(), BEST_MODEL_PATH)
        print(f' ---> Saved New Best Model (Val Acc: {best_acc:.4f}) to Drive.')
""")

add_code("""# 8. Final Test Evaluation & Visualization
print('\\n--- Final Test Set Evaluation ---')
if os.path.exists(BEST_MODEL_PATH):
    model.load_state_dict(torch.load(BEST_MODEL_PATH))
model.eval()

y_true, y_pred = [], []
with torch.no_grad():
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        _, preds = outputs.max(1)
        y_true.extend(labels.cpu().numpy())
        y_pred.extend(preds.cpu().numpy())

active_classes = [classes[i] for i in sorted(list(set(y_true)))]
if len(active_classes) > 0:
    print(classification_report(y_true, y_pred, target_names=active_classes, digits=4))
else:
    print("Warning: No predictions made!")

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(logs.get('train_loss', []), label='Train', marker='o')
plt.plot(logs.get('val_loss', []), label='Val', marker='o')
plt.title('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(logs.get('train_acc', []), label='Train', marker='o')
plt.plot(logs.get('val_acc', []), label='Val', marker='o')
plt.title('Accuracy')
plt.legend()
plt.savefig(os.path.join(EXP_DIR, 'training_curves.png'))
plt.show()

if len(active_classes) > 0:
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=False, cmap='Blues', xticklabels=active_classes, yticklabels=active_classes)
    plt.title('Confusion Matrix on Held-Out Test Set')
    plt.ylabel('True Class')
    plt.xlabel('Predicted Class')
    plt.savefig(os.path.join(EXP_DIR, 'confusion_matrix.png'))
    plt.show()
""")

notebook = {
  "cells": cells,
  "metadata": {
    "kernelspec": {
      "display_name": "Python 3",
      "language": "python",
      "name": "python3"
    }
  },
  "nbformat": 4,
  "nbformat_minor": 4
}

out_dir = r"C:\Projects\SmartPlate\ml\notebooks"
os.makedirs(out_dir, exist_ok=True)
with open(os.path.join(out_dir, "smartplate_food_classification.ipynb"), "w") as f:
    json.dump(notebook, f, indent=2)
print("Updated notebook generated successfully.")
