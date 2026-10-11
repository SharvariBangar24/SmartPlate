import nbformat as nbf
import os

nb = nbf.v4.new_notebook()

cells = []

cells.append(nbf.v4.new_markdown_cell("""# SmartPlate ONNX Export & Validation
This notebook exports the trained YOLO11s and MobileNetV3-Large models to ONNX format.
Ensure that both `best.pt` (YOLO) and `mobilenetv3_large_best_40classes.pt` are accessible in your Google Drive."""))

cells.append(nbf.v4.new_code_cell("""# Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')"""))

cells.append(nbf.v4.new_code_cell("""# Install required packages
!pip install -q ultralytics onnx onnxruntime torch torchvision numpy"""))

cells.append(nbf.v4.new_code_cell("""import os
import torch
import torch.nn as nn
import torchvision.models as models
import onnx
import onnxruntime as ort
import numpy as np
from ultralytics import YOLO
import shutil

# Paths
DRIVE_DIR = '/content/drive/MyDrive/Smartplate'
EXP_ONNX_DIR = os.path.join(DRIVE_DIR, 'experiments', 'onnx_export_1')
os.makedirs(EXP_ONNX_DIR, exist_ok=True)

YOLO_PT_PATH = os.path.join(DRIVE_DIR, 'models', 'best.pt')
MOBILENET_PT_PATH = os.path.join(DRIVE_DIR, 'experiments', 'experiment_classification_1', 'mobilenetv3_large_best_40classes.pt')

YOLO_ONNX_PATH = os.path.join(EXP_ONNX_DIR, 'best.onnx')
MOBILENET_ONNX_PATH = os.path.join(EXP_ONNX_DIR, 'mobilenetv3_40classes.onnx')
REPORT_PATH = os.path.join(EXP_ONNX_DIR, 'validation_report.txt')"""))

cells.append(nbf.v4.new_markdown_cell("""## 1. Export YOLO11s to ONNX"""))
cells.append(nbf.v4.new_code_cell("""if not os.path.exists(YOLO_PT_PATH):
    print(f"Error: YOLO checkpoint not found at {YOLO_PT_PATH}. Please upload it to your Drive.")
else:
    yolo_model = YOLO(YOLO_PT_PATH)
    # Export returns the path to the exported file
    exported_path = yolo_model.export(format='onnx', imgsz=640, dynamic=True)
    
    # YOLO automatically saves it alongside the .pt file. Let's move it.
    if os.path.exists(exported_path):
        shutil.copy(exported_path, YOLO_ONNX_PATH)
        print(f"YOLO exported successfully to {YOLO_ONNX_PATH}")
    else:
        print("Export failed.")"""))

cells.append(nbf.v4.new_markdown_cell("""## 2. Export MobileNetV3-Large to ONNX"""))
cells.append(nbf.v4.new_code_cell("""if not os.path.exists(MOBILENET_PT_PATH):
    print(f"Error: MobileNet checkpoint not found at {MOBILENET_PT_PATH}.")
else:
    model_cls = models.mobilenet_v3_large(weights=None)
    in_features = model_cls.classifier[3].in_features
    model_cls.classifier[3] = nn.Linear(in_features, 40)
    
    checkpoint = torch.load(MOBILENET_PT_PATH, map_location='cpu', weights_only=True)
    if 'model_state_dict' in checkpoint:
        model_cls.load_state_dict(checkpoint['model_state_dict'])
    else:
        model_cls.load_state_dict(checkpoint)
        
    model_cls.eval()
    
    dummy_input = torch.randn(1, 3, 224, 224)
    
    torch.onnx.export(
        model_cls,
        dummy_input,
        MOBILENET_ONNX_PATH,
        export_params=True,
        opset_version=14,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
    )
    print(f"MobileNetV3 exported successfully to {MOBILENET_ONNX_PATH}")"""))

cells.append(nbf.v4.new_markdown_cell("""## 3. Validation: PyTorch vs ONNXRuntime"""))
cells.append(nbf.v4.new_code_cell("""report_lines = ["ONNX Validation Report\\n", "="*30 + "\\n"]

def validate_mobilenet():
    if not os.path.exists(MOBILENET_ONNX_PATH):
        return "MobileNet ONNX not found."
    
    model_cls.eval()
    dummy_input = torch.randn(1, 3, 224, 224)
    with torch.no_grad():
        pt_out = model_cls(dummy_input).numpy()
        
    ort_session = ort.InferenceSession(MOBILENET_ONNX_PATH)
    ort_inputs = {ort_session.get_inputs()[0].name: dummy_input.numpy()}
    ort_out = ort_session.run(None, ort_inputs)[0]
    
    try:
        np.testing.assert_allclose(pt_out, ort_out, rtol=1e-03, atol=1e-05)
        return "MobileNetV3 PyTorch vs ONNX Match: SUCCESS (Max diff: {:.6f})".format(np.max(np.abs(pt_out - ort_out)))
    except AssertionError as e:
        return f"MobileNetV3 PyTorch vs ONNX Match: FAILED\\n{e}"

def validate_yolo():
    if not os.path.exists(YOLO_ONNX_PATH):
        return "YOLO ONNX not found."
    try:
        yolo_ort = YOLO(YOLO_ONNX_PATH)
        dummy_img = np.zeros((640, 640, 3), dtype=np.uint8)
        results = yolo_ort(dummy_img, verbose=False)
        return f"YOLO ONNX Smoke Test: SUCCESS. Found {len(results[0].boxes)} boxes on blank image."
    except Exception as e:
        return f"YOLO ONNX Smoke Test: FAILED\\n{e}"

mn_res = validate_mobilenet()
yo_res = validate_yolo()

print(mn_res)
print(yo_res)

report_lines.extend([mn_res + "\\n", yo_res + "\\n"])

with open(REPORT_PATH, 'w') as f:
    f.writelines(report_lines)
    
print(f"\\nReport saved to {REPORT_PATH}")"""))

nb['cells'] = cells

with open(r'c:\Projects\SmartPlate\ml\notebooks\smartplate_onnx_export.ipynb', 'w') as f:
    nbf.write(nb, f)
print("Notebook created.")
