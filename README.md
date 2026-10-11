# Smart Plate

Smart Plate is an intelligent dietary assessment system that combines computer vision and nutritional science to recognize foods, estimate portion weights, and calculate detailed macronutrients from plate images.

---

## Architecture Overview

1. **Food Detection:** YOLO11s exported to ONNX (`ml/experiments/onnx_export_1/best.onnx`) detects 14 primary food categories and localizes bounding boxes.
2. **Food Classification Refinement:** MobileNetV3 Large 40-class ONNX model (`ml/experiments/onnx_export_1/mobilenetv3_40classes.onnx`) refines broad categories into specific Indian dishes.
3. **Portion Estimation:** HistGradientBoosting Regressor (`ml/experiments/experiment_portion_1/models/best_portion_regressor.joblib`) estimates food weight in grams directly from bounding box geometry and food type.
4. **Nutritional Engine:** 54-dish reference database grounded in ICMR-NIN, IFCT 2017, and USDA FoodData Central calculates calories, protein, carbohydrates, and fat scaled to the estimated portion weight.
5. **Interactive UI:** Responsive glassmorphism web interface with real-time macro breakdowns and visual confidence badges.

---

## Project Structure

```text
SmartPlate/
├── backend/
│   └── app.py                     # Flask REST API server & inference engine
├── frontend/
│   ├── index.html                 # Web dashboard layout
│   ├── script.js                  # API caller & UI state manager
│   ├── style.css                  # Modern glassmorphism design system
│   ├── test_biryani.jpg           # Standard verification test image
│   └── assets/                    # UI images and icons
├── ml/
│   ├── evaluation/
│   │   └── nutrition/             # Phase A & B nutrition evaluation report, metrics, and charts
│   ├── experiments/
│   │   ├── onnx_export_1/         # Production ONNX models (best.onnx, mobilenetv3_40classes.onnx)
│   │   ├── experiment_classification_1/ # 40-class mappings (classes.json) and training metrics
│   │   ├── experiment_portion_1/  # Portion regressor model (best_portion_regressor.joblib) & report
│   │   ├── experiment_nutrition_2/# 7-experiment deep dive, improvement roadmap, and charts
│   │   └── experiment_image_validation_1/ # Confusion matrices and detection evaluation
│   └── scripts/                   # Integration tests and evaluation scripts
├── requirements.txt               # Minimal runtime Python dependencies
└── README.md                      # Project documentation
```

---

## Quickstart Guide

### 1. Prerequisites
- Python 3.10+
- A modern web browser (Chrome, Edge, Firefox)

### 2. Backend Setup
1. Clone the repository and navigate into the project directory:
   ```bash
   git clone https://github.com/SharvariBangar24/SmartPlate.git
   cd SmartPlate
   ```
2. Create and activate a virtual environment:
   ```powershell
   # Windows PowerShell
   python -m venv .venv
   .venv\Scripts\Activate.ps1
   ```
   ```bash
   # Linux / macOS
   python3 -m venv .venv
   source .venv/bin/activate
   ```
3. Install the minimal runtime dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Start the Flask backend server:
   ```bash
   python backend/app.py
   ```
   The backend starts at `http://127.0.0.1:5000`.

### 3. Frontend Setup
In a separate terminal, serve the frontend static files:
```powershell
python -m http.server 8000 --directory frontend
```
Open **`http://127.0.0.1:8000`** in your browser.

---

## Testing & Verification

1. On the web dashboard, upload `frontend/test_biryani.jpg`.
2. Click **Analyze Plate**.
3. Verify that the system displays:
   - Detected food item: **Biryani**
   - Estimated weight (g)
   - Calculated calories (kcal), protein (g), carbs (g), and fat (g)
   - Total plate nutrition summary cards.

You can also test the backend API directly:
```bash
curl -X POST http://127.0.0.1:5000/api/analyze -F "image=@frontend/test_biryani.jpg"
```

---

## Evaluation Reports & Metrics for Project Report

The following verified reports and visualization charts are ready for inclusion in the final project report and presentation:
- **Nutrition Evaluation Report:** `ml/evaluation/nutrition/NUTRITION_EVALUATION_REPORT.md`
- **Nutrition Accuracy Charts:** `ml/evaluation/nutrition/nutrition_accuracy_charts.png`
- **Summary Metrics JSON:** `ml/evaluation/nutrition/summary_metrics.json`
- **7-Experiment Nutritional Deep Dive:** `ml/experiments/experiment_nutrition_2/REPORT.md`
- **Consolidated Experiment Charts:** `ml/experiments/experiment_nutrition_2/charts/nutrition_experiment_charts.png`
- **Portion Estimation Benchmarks:** `ml/experiments/experiment_portion_1/REPORT.md`
