import os
import sys
import numpy as np
try:
    import onnxruntime as ort
except ImportError:
    print("Error: onnxruntime is not installed. Run: pip install onnxruntime")
    sys.exit(1)

def run_smoke_test():
    print("Starting ONNX Runtime Smoke Test...")
    
    # Paths to the ONNX models
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    onnx_dir = os.path.join(base_dir, 'experiments', 'onnx_export_1')
    
    yolo_onnx = os.path.join(onnx_dir, 'best.onnx')
    mobilenet_onnx = os.path.join(onnx_dir, 'mobilenetv3_40classes.onnx')
    
    # 1. Test YOLO11s ONNX
    if not os.path.exists(yolo_onnx):
        print(f"[FAIL] YOLO ONNX not found at {yolo_onnx}. Please download it from Drive.")
    else:
        try:
            ort_sess_yolo = ort.InferenceSession(yolo_onnx)
            input_name = ort_sess_yolo.get_inputs()[0].name
            # YOLO typical input shape: [1, 3, 640, 640], dtype float32
            dummy_yolo_input = np.random.randn(1, 3, 640, 640).astype(np.float32)
            out_yolo = ort_sess_yolo.run(None, {input_name: dummy_yolo_input})
            print(f"[PASS] YOLO ONNX loaded and executed. Output shape: {out_yolo[0].shape}")
        except Exception as e:
            print(f"[FAIL] YOLO ONNX execution error: {e}")

    # 2. Test MobileNetV3 ONNX
    if not os.path.exists(mobilenet_onnx):
        print(f"[FAIL] MobileNet ONNX not found at {mobilenet_onnx}. Please download it from Drive.")
    else:
        try:
            ort_sess_mn = ort.InferenceSession(mobilenet_onnx)
            input_name = ort_sess_mn.get_inputs()[0].name
            # MobileNet typical input shape: [1, 3, 224, 224], dtype float32
            dummy_mn_input = np.random.randn(1, 3, 224, 224).astype(np.float32)
            out_mn = ort_sess_mn.run(None, {input_name: dummy_mn_input})
            
            # Load class mapping to verify the outputs
            classes_path = os.path.join(base_dir, 'experiments', 'experiment_classification_1', 'classes.json')
            if os.path.exists(classes_path):
                import json
                with open(classes_path, 'r') as f:
                    class_mapping = json.load(f)
                predicted_class_id = str(np.argmax(out_mn[0]))
                predicted_class_name = class_mapping.get(predicted_class_id, "Unknown")
                print(f"[PASS] MobileNet ONNX loaded and executed. Output shape: {out_mn[0].shape}. Dummy predicted class: {predicted_class_name}")
            else:
                print(f"[PASS] MobileNet ONNX loaded. Output shape: {out_mn[0].shape}. (classes.json not found for mapping test)")
                
        except Exception as e:
            print(f"[FAIL] MobileNet ONNX execution error: {e}")

if __name__ == '__main__':
    run_smoke_test()
