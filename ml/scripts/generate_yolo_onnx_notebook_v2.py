import nbformat as nbf
import os

nb = nbf.v4.new_notebook()
cells = []

cells.append(nbf.v4.new_markdown_cell("""# YOLO11s ONNX Export & Validation
This notebook expects your original YOLO `best.pt` to be at `/content/drive/MyDrive/Smartplate/experiments/experiment_3_100epochs/weights/best.pt`.
It exports it to ONNX, runs inference on a real food image to mathematically validate it against PyTorch, and appends the result to the validation report."""))

cells.append(nbf.v4.new_code_cell("""# Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')"""))

cells.append(nbf.v4.new_code_cell("""# Install required packages
!pip install -q ultralytics onnx onnxruntime opencv-python-headless"""))

cells.append(nbf.v4.new_code_cell("""import os
import shutil
import numpy as np
import cv2
import urllib.request
from ultralytics import YOLO

DRIVE_DIR = '/content/drive/MyDrive/Smartplate'
EXP_ONNX_DIR = os.path.join(DRIVE_DIR, 'experiments', 'onnx_export_1')
REPORT_PATH = os.path.join(EXP_ONNX_DIR, 'validation_report.txt')

# We expect the file exactly here as per your manual upload.
EXPECTED_PT_PATH = os.path.join(DRIVE_DIR, 'experiments', 'experiment_3_100epochs', 'weights', 'best.pt')
YOLO_ONNX_PATH = os.path.join(EXP_ONNX_DIR, 'best.onnx')

os.makedirs(EXP_ONNX_DIR, exist_ok=True)

if not os.path.exists(EXPECTED_PT_PATH):
    raise FileNotFoundError(f"Checkpoint not found at {EXPECTED_PT_PATH}. Please ensure you uploaded it exactly to this location in Google Drive.")
else:
    print(f"Found YOLO checkpoint at: {EXPECTED_PT_PATH}")

# Export to ONNX
print("\\nExporting to ONNX...")
pt_model = YOLO(EXPECTED_PT_PATH)
exported_path = pt_model.export(format='onnx', imgsz=640, dynamic=True)

if os.path.exists(exported_path):
    shutil.copy(exported_path, YOLO_ONNX_PATH)
    print(f"Successfully copied exported ONNX to: {YOLO_ONNX_PATH}")
else:
    raise RuntimeError(f"ONNX export failed, file not found at {exported_path}")"""))

cells.append(nbf.v4.new_code_cell("""# Validation on a REAL food image
print("\\nValidating PyTorch vs ONNX Runtime on a real food image...")

# Download a real food image (Chicken Biryani)
sample_img_path = 'sample_food.jpg'
food_url = 'https://upload.wikimedia.org/wikipedia/commons/thumb/a/a8/Chicken_biryani.jpg/640px-Chicken_biryani.jpg'
urllib.request.urlretrieve(food_url, sample_img_path) 
img = cv2.imread(sample_img_path)

# PyTorch Inference
results_pt = pt_model(img, verbose=False)
pt_boxes = results_pt[0].boxes.xyxy.cpu().numpy()
pt_confs = results_pt[0].boxes.conf.cpu().numpy()
pt_classes = results_pt[0].boxes.cls.cpu().numpy()

# ONNX Inference via Ultralytics (automatically uses onnxruntime)
onnx_model = YOLO(YOLO_ONNX_PATH, task='detect')
results_onnx = onnx_model(img, verbose=False)
onnx_boxes = results_onnx[0].boxes.xyxy.cpu().numpy()
onnx_confs = results_onnx[0].boxes.conf.cpu().numpy()
onnx_classes = results_onnx[0].boxes.cls.cpu().numpy()

# Compare class IDs, confidences, and bounding boxes
try:
    if len(pt_boxes) != len(onnx_boxes):
        raise AssertionError(f"Detection count mismatch: PyTorch found {len(pt_boxes)}, ONNX found {len(onnx_boxes)}")
    
    # Sort boxes by confidence to align them for comparison
    pt_sort_idx = np.argsort(-pt_confs)
    onnx_sort_idx = np.argsort(-onnx_confs)
    
    pt_boxes_sorted = pt_boxes[pt_sort_idx]
    onnx_boxes_sorted = onnx_boxes[onnx_sort_idx]
    
    pt_confs_sorted = pt_confs[pt_sort_idx]
    onnx_confs_sorted = onnx_confs[onnx_sort_idx]
    
    pt_classes_sorted = pt_classes[pt_sort_idx]
    onnx_classes_sorted = onnx_classes[onnx_sort_idx]
    
    # Mathematical Validation
    np.testing.assert_equal(pt_classes_sorted, onnx_classes_sorted, err_msg="Class IDs do not match")
    np.testing.assert_allclose(pt_confs_sorted, onnx_confs_sorted, rtol=1e-2, atol=1e-2, err_msg="Confidences differ significantly")
    np.testing.assert_allclose(pt_boxes_sorted, onnx_boxes_sorted, rtol=1e-2, atol=2.0, err_msg="Bounding boxes differ significantly")
    
    validation_msg = f"YOLO11s ONNX vs PyTorch Real Image Match: SUCCESS. (Detected {len(pt_boxes)} identical objects)"
    print(validation_msg)
except Exception as e:
    validation_msg = f"YOLO11s ONNX vs PyTorch Real Image Match: FAILED\\nError: {e}"
    print(validation_msg)
    raise AssertionError(validation_msg)"""))

cells.append(nbf.v4.new_code_cell("""# Update Validation Report
print("\\nUpdating validation_report.txt...")

existing_content = ""
if os.path.exists(REPORT_PATH):
    with open(REPORT_PATH, 'r') as f:
        existing_content = f.read()

lines = existing_content.split('\\n')
new_lines = [line for line in lines if "YOLO ONNX Smoke Test" not in line and "YOLO11s ONNX vs PyTorch" not in line and line.strip() != ""]

# Append new result without destroying MobileNet records
new_lines.append(validation_msg)

with open(REPORT_PATH, 'w') as f:
    f.write('\\n'.join(new_lines) + '\\n')

print(f"Appended YOLO validation result to {REPORT_PATH}")"""))

nb['cells'] = cells

with open(r'c:\Projects\SmartPlate\ml\notebooks\fix_yolo_onnx_export.ipynb', 'w') as f:
    nbf.write(nb, f)
print("Notebook generator written.")
