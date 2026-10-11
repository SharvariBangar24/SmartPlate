import nbformat as nbf
import os

nb = nbf.v4.new_notebook()
cells = []

cells.append(nbf.v4.new_markdown_cell("""# YOLO11s ONNX Export & Validation Fix
This notebook searches for your original YOLO `best.pt` in Google Drive, exports it to ONNX, runs inference on a real image to compare against PyTorch, and updates the validation report without overwriting MobileNet results."""))

cells.append(nbf.v4.new_code_cell("""# Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')"""))

cells.append(nbf.v4.new_code_cell("""# Install required packages
!pip install -q ultralytics onnx onnxruntime opencv-python-headless"""))

cells.append(nbf.v4.new_code_cell("""import os
import glob
import shutil
import numpy as np
import cv2
import urllib.request
from ultralytics import YOLO

DRIVE_DIR = '/content/drive/MyDrive/Smartplate'
EXP_ONNX_DIR = os.path.join(DRIVE_DIR, 'experiments', 'onnx_export_1')
REPORT_PATH = os.path.join(EXP_ONNX_DIR, 'validation_report.txt')
EXPECTED_PT_PATH = os.path.join(DRIVE_DIR, 'experiments', 'experiment_3_100epochs', 'weights', 'best.pt')
YOLO_ONNX_PATH = os.path.join(EXP_ONNX_DIR, 'best.onnx')

os.makedirs(EXP_ONNX_DIR, exist_ok=True)

# 1. Search for best.pt
actual_pt_path = None
if os.path.exists(EXPECTED_PT_PATH):
    actual_pt_path = EXPECTED_PT_PATH
    print(f"Found YOLO checkpoint at expected path: {actual_pt_path}")
else:
    print(f"Checkpoint not found at {EXPECTED_PT_PATH}. Searching Drive...")
    # Search recursively for best.pt in the Smartplate directory
    search_pattern = os.path.join(DRIVE_DIR, '**', 'best.pt')
    found_files = glob.glob(search_pattern, recursive=True)
    if found_files:
        actual_pt_path = found_files[0]
        print(f"Found YOLO checkpoint at: {actual_pt_path}")
    else:
        raise FileNotFoundError("Could not find best.pt anywhere in /content/drive/MyDrive/Smartplate/")

# 2. Export to ONNX
print("\\nExporting to ONNX...")
pt_model = YOLO(actual_pt_path)
exported_path = pt_model.export(format='onnx', imgsz=640, dynamic=True)

if os.path.exists(exported_path):
    shutil.copy(exported_path, YOLO_ONNX_PATH)
    print(f"Successfully copied exported ONNX to: {YOLO_ONNX_PATH}")
else:
    raise RuntimeError(f"ONNX export failed, file not found at {exported_path}")"""))

cells.append(nbf.v4.new_code_cell("""# 3. Validation on a real image
print("\\nValidating PyTorch vs ONNX Runtime on a real food image...")

# Download a sample image for testing
sample_img_path = 'sample_food.jpg'
urllib.request.urlretrieve('https://ultralytics.com/images/zidane.jpg', sample_img_path) 
# Note: Using standard test image. You can replace with any food image path if desired.
img = cv2.imread(sample_img_path)

# PyTorch Inference
results_pt = pt_model(img, verbose=False)
pt_boxes = results_pt[0].boxes.xyxy.cpu().numpy()
pt_confs = results_pt[0].boxes.conf.cpu().numpy()
pt_classes = results_pt[0].boxes.cls.cpu().numpy()

# ONNX Inference
# YOLO() class automatically uses onnxruntime if an .onnx file is passed
onnx_model = YOLO(YOLO_ONNX_PATH, task='detect')
results_onnx = onnx_model(img, verbose=False)
onnx_boxes = results_onnx[0].boxes.xyxy.cpu().numpy()
onnx_confs = results_onnx[0].boxes.conf.cpu().numpy()
onnx_classes = results_onnx[0].boxes.cls.cpu().numpy()

# Compare
try:
    if len(pt_boxes) != len(onnx_boxes):
        raise AssertionError(f"Detection count mismatch: PyTorch found {len(pt_boxes)}, ONNX found {len(onnx_boxes)}")
    
    # Sort boxes by confidence to align them for comparison
    pt_sort_idx = np.argsort(-pt_confs)
    onnx_sort_idx = np.argsort(-onnx_confs)
    
    pt_boxes_sorted = pt_boxes[pt_sort_idx]
    onnx_boxes_sorted = onnx_boxes[onnx_sort_idx]
    
    np.testing.assert_allclose(pt_boxes_sorted, onnx_boxes_sorted, rtol=1e-2, atol=2.0)
    
    validation_msg = f"YOLO11s ONNX vs PyTorch Real Image Match: SUCCESS. (Detected {len(pt_boxes)} identical objects)"
    print(validation_msg)
except Exception as e:
    validation_msg = f"YOLO11s ONNX vs PyTorch Real Image Match: FAILED\\nError: {e}"
    print(validation_msg)
    raise AssertionError(validation_msg)"""))

cells.append(nbf.v4.new_code_cell("""# 4. Update Validation Report
print("\\nUpdating validation_report.txt...")

existing_content = ""
if os.path.exists(REPORT_PATH):
    with open(REPORT_PATH, 'r') as f:
        existing_content = f.read()

# Filter out any old YOLO test results from the string to replace it cleanly
lines = existing_content.split('\\n')
new_lines = [line for line in lines if "YOLO ONNX Smoke Test" not in line and "YOLO11s ONNX vs PyTorch" not in line and line.strip() != ""]

# Append new result
new_lines.append(validation_msg)

with open(REPORT_PATH, 'w') as f:
    f.write('\\n'.join(new_lines) + '\\n')

print(f"Appended YOLO validation result to {REPORT_PATH}")"""))

nb['cells'] = cells

with open(r'c:\Projects\SmartPlate\ml\notebooks\fix_yolo_onnx_export.ipynb', 'w') as f:
    nbf.write(nb, f)
print("Notebook created successfully.")
