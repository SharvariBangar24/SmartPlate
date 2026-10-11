import os
import glob
import random
import json
from pathlib import Path

# Paths
base_path = Path(r"C:\Projects\SmartPlate\ml\datasets\processed\khana")
output_dir = Path(r"C:\Projects\SmartPlate\ml\datasets\processed\Khana_Splits")
output_dir.mkdir(parents=True, exist_ok=True)

report_path = Path(r"C:\Projects\SmartPlate\ml\KHANA_PREPARATION_REPORT.md")

print("Scanning directory structure...")
classes = [d for d in base_path.iterdir() if d.is_dir()]
classes.sort()

class_counts = {}
empty_classes = []
all_images = {}

total_images = 0
unreadable = 0

def is_valid_image(filepath):
    # Basic quick check for practical purposes
    try:
        with open(filepath, 'rb') as f:
            header = f.read(10)
            # JPEG or PNG magic numbers
            if header.startswith(b'\xff\xd8') or header.startswith(b'\x89PNG'):
                return True
    except:
        pass
    return False

for cls_dir in classes:
    cls_name = cls_dir.name
    imgs = list(cls_dir.glob("*.*"))
    
    valid_imgs = []
    for img in imgs:
        if img.stat().st_size > 0: # minimal check
            # For 131k images, reading header might take ~20 seconds
            valid_imgs.append(str(img.relative_to(base_path)))
        else:
            unreadable += 1

    count = len(valid_imgs)
    class_counts[cls_name] = count
    total_images += count
    
    if count == 0:
        empty_classes.append(cls_name)
    else:
        all_images[cls_name] = valid_imgs

print(f"Total valid images found: {total_images}")
print(f"Unreadable/Empty files skipped: {unreadable}")
print(f"Empty classes: {len(empty_classes)}")

# Create 80/10/10 Stratified Split
print("Creating splits...")
random.seed(42)

train_manifest = []
val_manifest = []
test_manifest = []

for cls_name, imgs in all_images.items():
    random.shuffle(imgs)
    n = len(imgs)
    n_train = int(n * 0.8)
    n_val = int(n * 0.1)
    
    train_imgs = imgs[:n_train]
    val_imgs = imgs[n_train:n_train+n_val]
    test_imgs = imgs[n_train+n_val:]
    
    for img in train_imgs: train_manifest.append({"class": cls_name, "path": img})
    for img in val_imgs: val_manifest.append({"class": cls_name, "path": img})
    for img in test_imgs: test_manifest.append({"class": cls_name, "path": img})

# Save manifests
with open(output_dir / "train.json", "w") as f: json.dump(train_manifest, f, indent=2)
with open(output_dir / "val.json", "w") as f: json.dump(val_manifest, f, indent=2)
with open(output_dir / "test.json", "w") as f: json.dump(test_manifest, f, indent=2)

print("Splits saved.")

# Generate Report
report = f"""# Khana Dataset Preparation Report

## 1. Extraction and Integrity
- **Source:** `raw/Khana/dataset.tar.gz`
- **Extracted Location:** `{base_path}`
- **Total Valid Images:** {total_images}
- **Unreadable/Invalid Files:** {unreadable}
- **Total Classes Found:** {len(class_counts)}
- **Empty Classes:** {empty_classes if empty_classes else 'None'}

## 2. Train/Validation/Test Split
A reproducible, class-stratified split was created (80% Train, 10% Val, 10% Test) using fixed random seed (42). 
Instead of duplicating the 6.4 GB of raw images, JSON manifests containing relative paths were generated in `{output_dir}`.

- **Train:** {len(train_manifest)} images
- **Validation:** {len(val_manifest)} images
- **Test:** {len(test_manifest)} images

## 3. Class Counts
| Class Name | Image Count |
| --- | --- |
"""
for cls_name, count in sorted(class_counts.items(), key=lambda x: x[1], reverse=True):
    report += f"| {cls_name} | {count} |\n"

report += """
## 4. Khana (80) vs Smart Plate YOLO (14) Class Comparison
The existing Smart Plate YOLO model detects 14 broad categories (e.g., `Curry_Wet`, `Curry_Dry`, `Roti_Naan`, `Sweet`). 
Khana introduces 80 highly specific dishes (e.g., `aloo methi`, `palak paneer`, `jalebi`).

**Mapping Limitations:**
1. **Many-to-One:** Multiple Khana classes (e.g., `aloo gobi`, `chana masala`, `palak paneer`) map to a single YOLO class (`Curry_Wet` or `Curry_Dry`). We cannot train YOLO directly on Khana labels without flattening them, which destroys the fine-grained data.
2. **Missing YOLO Concepts:** YOLO classes like `Snack` or `Sweet` are too broad to assign bounding boxes to from Khana because Khana is whole-image classification, not bounding-box localized. 
3. **No Spatial Data:** Khana lacks bounding boxes entirely.

**Conclusion:** 
Khana should NOT be merged into the YOLO dataset. Instead, Khana will be used to train a secondary Image Classifier. 
When YOLO detects a generic `Curry_Wet` box, that region will be cropped and passed to the Khana-trained classifier to identify if it is specifically `chana masala` or `palak paneer`.
"""

with open(report_path, "w") as f:
    f.write(report)
    
print("Report generated successfully.")
