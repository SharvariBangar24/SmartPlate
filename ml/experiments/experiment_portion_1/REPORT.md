# Smart Plate - Portion Estimation Experiment 1 Report

## 1. Executive Summary & Objective
- **Objective:** Evaluate portion-weight estimation in grams using the Food Portion Benchmark (FPB) annotations.
- **Protocol:** Strict separation between training, validation, and held-out test splits.
- **Constraints:**
  - Zero image downloading (metadata-only parsing).
  - Only inference-time features used: bounding box width, height, area, aspect ratio, sqrt area, log area, and class ID.
  - No ground-truth weight, depth maps, or hidden physical sensors used as inputs.

## 2. Dataset Verification & Sample Counts
- **Source:** Institute of Smart Systems and Artificial Intelligence (ISSAI), Nazarbayev University (`issai/Food_Portion_Benchmark`).
- **License:** CC BY-NC 4.0.
- **Total Classes:** 138 Central Asian / Continental food categories.
- **Filtering Rule:** Records with `weight <= 0` (unmeasured secondary items) were strictly discarded.

| Split | Raw Bounding Boxes | Discarded (`weight <= 0`) | Usable Samples | Unique Images | Mean Weight | Median Weight | Std Dev |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Train** | 15,348 | 1,592 | **13,756** | 9,178 | 196.7 g | 166.0 g | 145.5 g |
| **Validation** | 5,943 | 220 | **5,723** | 2,351 | 220.9 g | 192.7 g | 144.7 g |
| **Test** | 2,784 | 122 | **2,662** | 2,173 | 205.1 g | 173.8 g | 143.7 g |

## 3. Model Comparison & Metrics

| Model Architecture | Validation MAE (g) | Validation RMSE (g) | Validation $R^2$ | Test MAE (g) | Test RMSE (g) | Test $R^2$ |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **A. Training Median Baseline** | 112.61 | 154.766 | -0.1437 | 107.989 | 148.856 | -0.074 |
| **B. Class-Conditional Mean** | 87.646 | 118.541 | 0.329 | 77.18 | 101.502 | 0.5006 |
| **C1. HistGradientBoosting** | **75.52** | **105.905** | **0.4644** | **61.61** | **87.476** | **0.6291** |
| **C2. Random Forest** | 93.365 | 126.415 | 0.2369 | 70.807 | 92.786 | 0.5827 |

- **Best Validation Model:** `HistGradientBoosting` (Val MAE: **75.52 g**).
- **Final Held-Out Test Split Performance:**
  - **MAE:** **61.61 grams**
  - **RMSE:** **87.476 grams**
  - **$R^2$ Score:** **0.6291**

## 4. Comparison to Published Benchmark Baseline
- The published IEEE Access baseline (Sanatbyek et al., 2025) reports an MAE of **90.95 g** using a full multi-task YOLOv12 model with an integrated weight regression head trained on raw RGB imagery at 640×640.
- Our lightweight geometric model `HistGradientBoosting` achieves an MAE of **61.61 g** on the test split using only 2D bounding-box geometry and class ID, without requiring GPU training or multi-gigabyte image downloads.

## 5. Domain & Generalization Limitations (CRITICAL)
1. **Culinary Domain Gap:** FPB was collected in Kazakhstan and reflects Central Asian cuisine (plov, shashlyk, beshbarmak, strudel, borsh).
2. **Indian Food Discrepancy:** The FPB portion models **CANNOT** be claimed to generalize to Indian dishes (e.g., wet vs dry curries, dal, sambar, biryani, roti) without recalibration on Indian food density and serving distributions.
3. **Camera Geometry & Reference Markers:** In uncalibrated smartphone captures, bounding box area varies with camera distance and angle unless a known reference plate/coin or volumetric sensor is present.

## 6. Artifact Inventory
- Code: `train_portion_models.py`
- Best Checkpoint: `models/best_portion_regressor.joblib`
- Dataset Summary: `dataset_summary.json`
- Configuration: `config.json`
- Metrics: `metrics.json`
- Test Sample Predictions: `test_predictions.csv`
- Visualization: `comparison_chart.png`
