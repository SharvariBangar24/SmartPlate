import os
import re
import json
from flask import Flask, jsonify, request
from flask_cors import CORS
from werkzeug.utils import secure_filename
import cv2
import numpy as np
import pandas as pd
import joblib
import onnxruntime as ort

app = Flask(__name__)
CORS(app)

# Application configuration
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16 MB limit for uploads

# Paths to models and metadata
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
YOLO_MODEL_PATH = os.path.join(BASE_DIR, 'ml', 'experiments', 'onnx_export_1', 'best.onnx')
MOBILENET_MODEL_PATH = os.path.join(BASE_DIR, 'ml', 'experiments', 'onnx_export_1', 'mobilenetv3_40classes.onnx')
CLASSES_PATH = os.path.join(BASE_DIR, 'ml', 'experiments', 'experiment_classification_1', 'classes.json')
PORTION_MODEL_PATH = os.path.join(BASE_DIR, 'ml', 'experiments', 'experiment_portion_1', 'models', 'best_portion_regressor.joblib')

# 14 YOLO detection classes (from dataset taxonomy / smartplate_combined data.yaml)
YOLO_CLASSES = [
    "Rice", "Biryani", "Roti_Naan", "Poori", "Dal_Sambar",
    "Curry_Wet", "Curry_Dry", "Idli", "Vada", "Chutney",
    "Curd", "Salad", "Egg", "Fish"
]

# Load MobileNetV3 40-class dictionary
CLASSIFIER_CLASSES = {}
try:
    if os.path.exists(CLASSES_PATH):
        with open(CLASSES_PATH, 'r', encoding='utf-8') as f:
            CLASSIFIER_CLASSES = json.load(f)
        print(f"Loaded {len(CLASSIFIER_CLASSES)} classifier classes from {CLASSES_PATH}")
    else:
        print(f"Warning: classes.json not found at {CLASSES_PATH}")
except Exception as e:
    print(f"Error loading classes.json: {e}")

# Conservative class refinement mapping: associates YOLO coarse categories to compatible fine-grained classes.
# Note: Specific categories (e.g. Biryani, Idli, Vada) are intentionally omitted because they must NOT be refined or overwritten.
CATEGORY_REFINEMENT_MAP = {
    "rice": {"white_rice", "fried_rice", "poha"},
    "roti_naan": {"roti_naan", "butter_naan"},
    "dal_sambar": {"dal_sambar", "chole", "rajma_curry"},
    "curry_wet": {
        "curry_wet", "shahi_paneer", "kadai_paneer", "fish_curry",
        "mutton_curry", "dum_aloo", "rajma_curry", "chole"
    },
    "curry_dry": {"aloo_gobi", "aloo_masala", "bhindi_masala"},
    "chutney": {"coconut_chutney", "green_chutney"}
}

# --- NUTRITION DATABASE ---
# Sources: Indian Food Composition Tables (IFCT, ICMR) & USDA FoodData Central
# Values represent standard per-serving nutritional estimates.
# Note: Portion-weight regression is in active development; these are documented serving-size estimates.
NUTRITION_DB = {
    # 40 Fine-grained classification dishes
    "aloo_gobi": {"serving": "1 portion (150g)", "cal": 130, "pro": 3.0, "carb": 16.0, "fat": 6.0},
    "aloo_masala": {"serving": "1 portion (150g)", "cal": 170, "pro": 3.5, "carb": 24.0, "fat": 7.0},
    "bhindi_masala": {"serving": "1 portion (150g)", "cal": 120, "pro": 2.5, "carb": 14.0, "fat": 6.0},
    "biryani": {"serving": "1 portion (300g)", "cal": 450, "pro": 15.0, "carb": 55.0, "fat": 18.0},
    "butter_naan": {"serving": "1 piece (80g)", "cal": 260, "pro": 6.5, "carb": 42.0, "fat": 7.5},
    "chai": {"serving": "1 cup (150ml)", "cal": 90, "pro": 2.5, "carb": 12.0, "fat": 3.5},
    "chole": {"serving": "1 bowl (150g)", "cal": 210, "pro": 9.0, "carb": 30.0, "fat": 6.0},
    "chole_bhature": {"serving": "1 plate (250g)", "cal": 480, "pro": 12.0, "carb": 62.0, "fat": 20.0},
    "coconut_chutney": {"serving": "2 tbsp (30g)", "cal": 50, "pro": 0.8, "carb": 3.0, "fat": 4.0},
    "curry_wet": {"serving": "1 bowl (150g)", "cal": 180, "pro": 6.0, "carb": 15.0, "fat": 10.0},
    "dal_sambar": {"serving": "1 bowl (150g)", "cal": 130, "pro": 7.0, "carb": 20.0, "fat": 3.0},
    "dhokla": {"serving": "2 pieces (80g)", "cal": 130, "pro": 4.0, "carb": 22.0, "fat": 3.0},
    "dum_aloo": {"serving": "1 bowl (150g)", "cal": 190, "pro": 3.5, "carb": 24.0, "fat": 9.0},
    "fish_curry": {"serving": "1 bowl (150g)", "cal": 210, "pro": 18.0, "carb": 6.0, "fat": 12.0},
    "fried_rice": {"serving": "1 portion (200g)", "cal": 280, "pro": 5.5, "carb": 45.0, "fat": 9.0},
    "ghevar": {"serving": "1 piece (50g)", "cal": 220, "pro": 2.0, "carb": 30.0, "fat": 10.0},
    "green_chutney": {"serving": "2 tbsp (30g)", "cal": 25, "pro": 0.8, "carb": 3.5, "fat": 0.8},
    "gulab_jamun": {"serving": "2 pieces (70g)", "cal": 300, "pro": 4.0, "carb": 46.0, "fat": 11.0},
    "idli": {"serving": "2 pieces (100g)", "cal": 120, "pro": 4.0, "carb": 24.0, "fat": 0.5},
    "jalebi": {"serving": "3 pieces (60g)", "cal": 240, "pro": 1.5, "carb": 42.0, "fat": 7.5},
    "kaathi_rolls": {"serving": "1 roll (150g)", "cal": 320, "pro": 11.0, "carb": 38.0, "fat": 14.0},
    "kadai_paneer": {"serving": "1 bowl (150g)", "cal": 260, "pro": 12.0, "carb": 10.0, "fat": 19.0},
    "kebab": {"serving": "2 pieces (100g)", "cal": 200, "pro": 18.0, "carb": 4.0, "fat": 12.0},
    "kheer": {"serving": "1 small bowl (100g)", "cal": 180, "pro": 4.5, "carb": 26.0, "fat": 6.5},
    "kulfi": {"serving": "1 stick (70g)", "cal": 160, "pro": 3.5, "carb": 20.0, "fat": 7.5},
    "lassi": {"serving": "1 glass (200ml)", "cal": 170, "pro": 6.0, "carb": 24.0, "fat": 5.5},
    "masala_dosa": {"serving": "1 dosa (180g)", "cal": 310, "pro": 6.0, "carb": 48.0, "fat": 10.0},
    "modak": {"serving": "2 pieces (60g)", "cal": 210, "pro": 2.5, "carb": 34.0, "fat": 7.0},
    "momos": {"serving": "6 pieces (150g)", "cal": 220, "pro": 8.0, "carb": 32.0, "fat": 6.0},
    "mutton_curry": {"serving": "1 bowl (150g)", "cal": 290, "pro": 21.0, "carb": 5.0, "fat": 20.0},
    "paani_puri": {"serving": "6 puris (120g)", "cal": 180, "pro": 3.5, "carb": 32.0, "fat": 4.5},
    "pakode": {"serving": "4 pieces (80g)", "cal": 220, "pro": 4.5, "carb": 22.0, "fat": 12.5},
    "pav_bhaji": {"serving": "1 plate (250g)", "cal": 380, "pro": 8.5, "carb": 54.0, "fat": 14.0},
    "poha": {"serving": "1 bowl (150g)", "cal": 190, "pro": 3.5, "carb": 33.0, "fat": 5.0},
    "rajma_curry": {"serving": "1 bowl (150g)", "cal": 190, "pro": 8.5, "carb": 28.0, "fat": 4.5},
    "ras_malai": {"serving": "2 pieces (80g)", "cal": 240, "pro": 6.0, "carb": 30.0, "fat": 10.0},
    "roti_naan": {"serving": "1 piece (40g)", "cal": 120, "pro": 3.5, "carb": 20.0, "fat": 2.0},
    "samosa": {"serving": "1 piece (80g)", "cal": 210, "pro": 3.5, "carb": 24.0, "fat": 11.0},
    "shahi_paneer": {"serving": "1 bowl (150g)", "cal": 280, "pro": 11.0, "carb": 12.0, "fat": 21.0},
    "white_rice": {"serving": "1 cup (158g)", "cal": 205, "pro": 4.3, "carb": 44.5, "fat": 0.4},

    # Coarse detection fallbacks
    "rice": {"serving": "1 cup (158g)", "cal": 205, "pro": 4.3, "carb": 44.5, "fat": 0.4},
    "poori": {"serving": "1 piece (30g)", "cal": 100, "pro": 2.0, "carb": 12.0, "fat": 5.0},
    "dal": {"serving": "1 bowl (150g)", "cal": 130, "pro": 7.0, "carb": 20.0, "fat": 3.0},
    "sambar": {"serving": "1 bowl (150g)", "cal": 130, "pro": 7.0, "carb": 20.0, "fat": 3.0},
    "curry_dry": {"serving": "1 portion (100g)", "cal": 150, "pro": 3.0, "carb": 18.0, "fat": 8.0},
    "vada": {"serving": "1 piece (40g)", "cal": 120, "pro": 4.0, "carb": 14.0, "fat": 6.0},
    "chutney": {"serving": "2 tbsp (30g)", "cal": 40, "pro": 1.0, "carb": 4.0, "fat": 2.0},
    "curd": {"serving": "1 small bowl (100g)", "cal": 60, "pro": 3.0, "carb": 4.0, "fat": 3.0},
    "salad": {"serving": "1 portion (100g)", "cal": 20, "pro": 1.0, "carb": 4.0, "fat": 0.0},
    "sweet": {"serving": "1 piece (40g)", "cal": 150, "pro": 2.0, "carb": 25.0, "fat": 5.0},
    "snack": {"serving": "1 portion (50g)", "cal": 150, "pro": 3.0, "carb": 15.0, "fat": 8.0},
    "egg": {"serving": "1 boiled (50g)", "cal": 68, "pro": 5.5, "carb": 0.6, "fat": 4.8},
    "fish": {"serving": "1 piece (100g)", "cal": 150, "pro": 18.0, "carb": 0.0, "fat": 8.0},
    "unknown": {"serving": "Estimate (100g)", "cal": 100, "pro": 5.0, "carb": 10.0, "fat": 5.0}
}

# Load ONNX Runtime Sessions on CPU
yolo_session = None
classifier_session = None

try:
    if os.path.exists(YOLO_MODEL_PATH):
        yolo_session = ort.InferenceSession(YOLO_MODEL_PATH, providers=['CPUExecutionProvider'])
        print(f"Loaded YOLO ONNX model from {YOLO_MODEL_PATH}")
    else:
        print(f"Warning: YOLO ONNX model not found at {YOLO_MODEL_PATH}")
except Exception as e:
    print(f"Error initializing YOLO ONNX session: {e}")

try:
    if os.path.exists(MOBILENET_MODEL_PATH):
        classifier_session = ort.InferenceSession(MOBILENET_MODEL_PATH, providers=['CPUExecutionProvider'])
        print(f"Loaded MobileNetV3 ONNX model from {MOBILENET_MODEL_PATH}")
    else:
        print(f"Warning: MobileNetV3 ONNX model not found at {MOBILENET_MODEL_PATH}")
except Exception as e:
    print(f"Error initializing MobileNetV3 ONNX session: {e}")

# Load portion estimation model
portion_model = None
try:
    if os.path.exists(PORTION_MODEL_PATH):
        portion_model = joblib.load(PORTION_MODEL_PATH)
        print(f"Loaded portion estimation model from {PORTION_MODEL_PATH}")
    else:
        print(f"Warning: Portion model not found at {PORTION_MODEL_PATH}")
except Exception as e:
    print(f"Error initializing portion model: {e}")

# Mapping from Smart Plate food categories to counterpart FPB dataset class indices (138 classes in FPB data.yaml)
# Unmapped classes smoothly default to NaN (HistGradientBoosting native missing category handler)
FPB_CLASS_MAP = {
    # Rice / Grain dishes
    "biryani": 98,          # plov
    "rice": 107,            # rice
    "white_rice": 107,      # rice
    "fried_rice": 107,      # rice
    "poha": 107,            # rice
    # Flatbreads
    "roti_naan": 16,        # bread
    "butter_naan": 16,      # bread
    "poori": 16,            # bread
    # Curries / Wet Legumes
    "dal_sambar": 75,       # lentiil_soup
    "dal": 75,              # lentiil_soup
    "sambar": 75,           # lentiil_soup
    "curry_wet": 75,        # lentiil_soup
    "chole": 75,            # lentiil_soup
    "rajma_curry": 75,      # lentiil_soup
    "shahi_paneer": 37,     # cottage_cheese
    "kadai_paneer": 37,     # cottage_cheese
    "mutton_curry": 6,      # beef
    "fish_curry": 47,       # fish_soup
    "fish": 111,            # salmon
    # Dry Curries / Vegetables
    "curry_dry": 57,        # grilled_vegetables
    "bhindi_masala": 57,    # grilled_vegetables
    "aloo_gobi": 57,        # grilled_vegetables
    "aloo_masala": 57,      # grilled_vegetables
    "dum_aloo": 57,         # grilled_vegetables
    # Salads / Dairy / Sides
    "salad": 49,            # fresh_salad
    "egg": 44,              # egg
    "curd": 63,             # kefir
    "lassi": 63,            # kefir
    "chutney": 65,          # ketchup
    "coconut_chutney": 65,  # ketchup
    "green_chutney": 65,    # ketchup
    # Snacks
    "samosa": 112,          # samsa
    "kebab": 117,           # shashlyk_chicken
    "french_fries": 48,     # french_fries
    "chai": 126             # tea_with_milk
}


def parse_standard_serving_weight(nutri_entry):
    """
    Extracts the standard reference weight in grams from serving string (e.g. '1 portion (150g)' -> 150.0).
    Falls back to 100.0g if not parseable.
    """
    serving_str = nutri_entry.get("serving", "")
    m = re.search(r'\((\d+)\s*(?:g|ml)\)', serving_str)
    if m:
        return float(m.group(1))
    return 100.0


def estimate_food_portion(bbox, img_width, img_height, food_name, standard_serving_g):
    """
    Estimates food portion weight in grams using the trained FPB HistGradientBoosting model.

    Expected model feature input order (from train_portion_models.py):
    ['class_id', 'w', 'h', 'area', 'aspect_ratio', 'sqrt_area', 'log_area']

    Safeguards:
    - If portion_model is unavailable -> return standard serving weight.
    - If bounding box is invalid/degenerate or image dimensions <= 0 -> return standard serving weight.
    - If model prediction is NaN, inf, or <= 0 -> fallback to standard serving weight.
    - If model prediction exceeds plausible single-plate portion bounds (< 15g or > 1200g) -> clamp.

    Returns:
    - (weight_g, method_str, is_directly_measured_weight=False)
    """
    if bbox is None or len(bbox) != 4:
        return standard_serving_g, "standard_serving_fallback_malformed_bbox", False

    x1, y1, x2, y2 = bbox
    bw = max(0.0, float(x2 - x1))
    bh = max(0.0, float(y2 - y1))

    if img_width <= 0 or img_height <= 0 or bw <= 0 or bh <= 0:
        return standard_serving_g, "standard_serving_fallback_invalid_geometry", False

    # Inference-time normalized geometric features (identical to train_portion_models.py)
    w_norm = min(1.0, max(0.001, bw / float(img_width)))
    h_norm = min(1.0, max(0.001, bh / float(img_height)))
    area = w_norm * h_norm
    aspect_ratio = w_norm / (h_norm + 1e-6)
    sqrt_area = float(np.sqrt(max(0.0, area)))
    log_area = float(np.log1p(max(0.0, area)))

    food_key = food_name.lower().replace(" ", "_")
    fpb_class_id = FPB_CLASS_MAP.get(food_key, np.nan)

    if portion_model is None:
        return standard_serving_g, "standard_serving_reference_no_model", False

    try:
        feature_dict = {
            'class_id': [fpb_class_id],
            'w': [w_norm],
            'h': [h_norm],
            'area': [area],
            'aspect_ratio': [aspect_ratio],
            'sqrt_area': [sqrt_area],
            'log_area': [log_area]
        }
        df_features = pd.DataFrame(feature_dict)
        raw_pred = float(portion_model.predict(df_features)[0])

        if np.isnan(raw_pred) or np.isinf(raw_pred) or raw_pred <= 0.0:
            return standard_serving_g, "standard_serving_fallback_invalid_prediction", False

        # Plausible portion bounds for single food item on a plate (15g to 1200g)
        clamped_pred = max(15.0, min(1200.0, raw_pred))
        return round(clamped_pred, 1), "fpb_geometric_regressor", False
    except Exception as e:
        print(f"Portion prediction exception: {e}")
        return standard_serving_g, "standard_serving_fallback_exception", False


def letterbox(img, target_shape=(640, 640), color=(114, 114, 114)):
    """
    Resizes image maintaining aspect ratio and pads with solid color to target_shape.
    """
    h, w = img.shape[:2]
    target_w, target_h = target_shape
    scale = min(target_w / w, target_h / h)
    nw, nh = int(round(w * scale)), int(round(h * scale))
    pad_w = (target_w - nw) / 2
    pad_h = (target_h - nh) / 2

    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    top = int(round(pad_h - 0.1))
    bottom = target_h - nh - top
    left = int(round(pad_w - 0.1))
    right = target_w - nw - left

    padded = cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return padded, scale, left, top


def run_yolo_detection(img, conf_thresh=0.25, iou_thresh=0.45):
    """
    Runs YOLO11s ONNX inference on image and applies Non-Maximum Suppression (NMS).
    """
    if yolo_session is None:
        raise RuntimeError("YOLO ONNX model session is not loaded.")

    h0, w0 = img.shape[:2]
    padded, scale, left, top = letterbox(img, (640, 640))

    # Preprocess: BGR -> RGB, Transpose to CHW, Normalize to [0, 1]
    blob = padded[:, :, ::-1].transpose(2, 0, 1)[np.newaxis, :, :, :].astype(np.float32) / 255.0

    input_name = yolo_session.get_inputs()[0].name
    output = yolo_session.run(None, {input_name: blob})[0]  # shape: (1, 18, 8400)

    # Transpose to (8400, 18) -> [cx, cy, w, h, s0..s13]
    preds = output[0].T
    boxes = preds[:, :4]
    scores = preds[:, 4:]

    max_scores = np.max(scores, axis=1)
    class_ids = np.argmax(scores, axis=1)

    mask = max_scores >= conf_thresh
    filt_boxes = boxes[mask]
    filt_scores = max_scores[mask]
    filt_classes = class_ids[mask]

    if len(filt_boxes) == 0:
        return []

    # Format boxes for cv2.dnn.NMSBoxes: [x, y, w, h] in canvas coordinates
    cv_boxes = []
    for b in filt_boxes:
        cx, cy, bw, bh = b
        bx = int(cx - bw / 2)
        by = int(cy - bh / 2)
        cv_boxes.append([bx, by, int(bw), int(bh)])

    indices = cv2.dnn.NMSBoxes(cv_boxes, filt_scores.tolist(), conf_thresh, iou_thresh)

    if len(indices) == 0:
        return []

    indices = np.array(indices).flatten()
    detections = []

    for idx in indices:
        b = filt_boxes[idx]
        cx, cy, bw, bh = b
        # Unscale from padded canvas back to original image coordinates
        x1 = (cx - bw / 2 - left) / scale
        y1 = (cy - bh / 2 - top) / scale
        x2 = (cx + bw / 2 - left) / scale
        y2 = (cy + bh / 2 - top) / scale

        # Clamp coordinates to original image bounds
        x1 = max(0.0, min(float(w0), float(x1)))
        y1 = max(0.0, min(float(h0), float(y1)))
        x2 = max(0.0, min(float(w0), float(x2)))
        y2 = max(0.0, min(float(h0), float(y2)))

        cls_id = int(filt_classes[idx])
        conf = float(filt_scores[idx])
        cls_name = YOLO_CLASSES[cls_id] if cls_id < len(YOLO_CLASSES) else f"class_{cls_id}"

        detections.append({
            "class_id": cls_id,
            "yolo_name": cls_name,
            "confidence": conf,
            "bbox": [round(x1, 1), round(y1, 1), round(x2, 1), round(y2, 1)]
        })

    return detections


def classify_crop(crop_img, yolo_class_name, yolo_conf):
    """
    Runs MobileNetV3-Large 40-class classification on cropped food region and
    conservatively refines the YOLO detection IF AND ONLY IF:
    1. The YOLO detection belongs to an explicitly supported coarse category.
    2. The classifier predicts a compatible fine-grained class from that category's allowed set.
    3. The classifier confidence meets the minimum threshold (>= 0.45).
    Otherwise, the original YOLO prediction and confidence are strictly preserved.
    """
    if classifier_session is None or crop_img is None or crop_img.size == 0:
        return yolo_class_name, yolo_conf

    h, w = crop_img.shape[:2]
    if h < 10 or w < 10:
        return yolo_class_name, yolo_conf

    # Check if this YOLO category is an explicitly supported coarse category
    yolo_key = yolo_class_name.lower().replace(" ", "_")
    allowed_candidates = CATEGORY_REFINEMENT_MAP.get(yolo_key)
    if not allowed_candidates:
        # Specific dishes (e.g. Biryani, Idli) are never refined by the secondary classifier
        return yolo_class_name, yolo_conf

    try:
        # Preprocess matching ImageNet transforms (RGB, 224x224, standard normalization)
        resized = cv2.resize(crop_img, (224, 224), interpolation=cv2.INTER_LINEAR)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        normalized = (rgb - mean) / std
        blob = normalized.transpose(2, 0, 1)[np.newaxis, :, :, :].astype(np.float32)

        input_name = classifier_session.get_inputs()[0].name
        logits = classifier_session.run(None, {input_name: blob})[0][0]

        # Numerically stable softmax
        exp_logits = np.exp(logits - np.max(logits))
        probs = exp_logits / np.sum(exp_logits)

        top_idx = int(np.argmax(probs))
        top_name = CLASSIFIER_CLASSES.get(str(top_idx), CLASSIFIER_CLASSES.get(top_idx, "unknown"))
        top_conf = float(probs[top_idx])

        # Only refine if the predicted fine-grained class is explicitly compatible and confident
        if top_name in allowed_candidates and top_conf >= 0.45:
            return top_name, top_conf

        # Default: preserve coarse YOLO detection
        return yolo_class_name, yolo_conf

    except Exception as e:
        print(f"Classifier crop inference exception: {e}")
        return yolo_class_name, yolo_conf


@app.route('/api/test', methods=['GET'])
def test_endpoint():
    return jsonify({
        "message": "Smart Plate backend is working!",
        "yolo_loaded": yolo_session is not None,
        "classifier_loaded": classifier_session is not None,
        "portion_regressor_loaded": portion_model is not None,
        "classes_count": len(CLASSIFIER_CLASSES)
    })


@app.route('/api/analyze', methods=['POST'])
def analyze_image():
    if yolo_session is None:
        return jsonify({"error": "Detection model (YOLO ONNX) is not loaded on the server."}), 500

    if 'image' not in request.files:
        return jsonify({"error": "No image file provided."}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({"error": "Empty filename."}), 400

    try:
        # Decode uploaded image bytes via OpenCV
        file_bytes = np.frombuffer(file.read(), np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        if img is None:
            return jsonify({"error": "Invalid image file."}), 400

        height, width, _ = img.shape

        # Step 1: Detect food regions via YOLO11s ONNX
        raw_detections = run_yolo_detection(img, conf_thresh=0.25, iou_thresh=0.45)

        processed_foods = []
        total_nutrition = {
            "calories": 0,
            "protein": 0.0,
            "carbs": 0.0,
            "fat": 0.0,
            "estimated_weight_g": 0.0
        }

        # Step 2: Crop, classify, and estimate portion for each detected food region
        for det in raw_detections:
            x1, y1, x2, y2 = det["bbox"]
            ix1, iy1, ix2, iy2 = int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2))

            crop = img[iy1:iy2, ix1:ix2]
            final_name, final_conf = classify_crop(crop, det["yolo_name"], det["confidence"])

            # Step 3: Lookup baseline nutrition information
            db_key = final_name.lower().replace(" ", "_")
            nutri_info = NUTRITION_DB.get(db_key)
            if nutri_info is None:
                # Secondary lookup with coarse YOLO key
                fallback_key = det["yolo_name"].lower().replace(" ", "_")
                nutri_info = NUTRITION_DB.get(fallback_key, NUTRITION_DB["unknown"])

            standard_serving_g = parse_standard_serving_weight(nutri_info)

            # Step 4: Estimate portion weight via 2D geometric regressor
            est_weight_g, method_str, is_measured = estimate_food_portion(
                det["bbox"], width, height, final_name, standard_serving_g
            )

            # Step 5: Scale nutritional values proportionally based on estimated weight
            scale_factor = est_weight_g / standard_serving_g if standard_serving_g > 0 else 1.0

            scaled_cal = round(nutri_info["cal"] * scale_factor)
            scaled_pro = round(nutri_info["pro"] * scale_factor, 1)
            scaled_carb = round(nutri_info["carb"] * scale_factor, 1)
            scaled_fat = round(nutri_info["fat"] * scale_factor, 1)

            total_nutrition["calories"] += scaled_cal
            total_nutrition["protein"] += scaled_pro
            total_nutrition["carbs"] += scaled_carb
            total_nutrition["fat"] += scaled_fat
            total_nutrition["estimated_weight_g"] += est_weight_g

            processed_foods.append({
                "class_id": det["class_id"],
                "name": final_name,
                "confidence": round(final_conf, 3),
                "bbox": det["bbox"],
                "nutrition": {
                    "serving_assumption": nutri_info["serving"],
                    "standard_serving_weight_g": standard_serving_g,
                    "estimated_weight_g": round(est_weight_g, 1),
                    "weight_method": method_str,
                    "is_directly_measured_weight": is_measured,
                    "calories": scaled_cal,
                    "protein": scaled_pro,
                    "carbs": scaled_carb,
                    "fat": scaled_fat
                }
            })

        return jsonify({
            "image_width": width,
            "image_height": height,
            "detection_count": len(processed_foods),
            "foods": processed_foods,
            "total_nutrition": {
                "estimated_weight_g": round(total_nutrition["estimated_weight_g"], 1),
                "is_directly_measured_weight": False,
                "calories": round(total_nutrition["calories"]),
                "protein": round(total_nutrition["protein"], 1),
                "carbs": round(total_nutrition["carbs"], 1),
                "fat": round(total_nutrition["fat"], 1)
            },
            "disclaimer": "Nutritional values are calculated using 2D geometric portion-weight regression (FPB model) scaled against standard nutritional density (IFCT/USDA). Portion estimates are non-volumetric approximations and have not been calibrated with measured ground-truth scales for Indian foods. Weight values are estimates, not directly measured."
        })

    except Exception as e:
        print(f"Inference error: {e}")
        return jsonify({"error": f"An error occurred during image analysis: {str(e)}"}), 500


@app.errorhandler(413)
def request_entity_too_large(error):
    return jsonify({"error": "File size exceeds the 16MB limit."}), 413


if __name__ == '__main__':
    app.run(debug=True)
