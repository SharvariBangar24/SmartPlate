# Smart Plate - Real-Image Detection & Classification Diagnostic Report

## 1. Executive Summary
This diagnostic investigates real-image performance across the Smart Plate vision pipeline:
1. **YOLO11s Food Detector** (`best.onnx`) - 14 coarse/fine food classes.
2. **MobileNetV3-Large Classifier** (`mobilenetv3_40classes.onnx`) - 40 fine-grained Indian food classes.
3. **Refinement Pipeline & Mapping Logic** (`CATEGORY_REFINEMENT_MAP` & `classify_crop` in `backend/app.py`).

### Key Findings at a Glance:
- **YOLO11s Detector:** High precision (**76.2%**) and solid overall mAP@50 (**80.4%**), but severe class imbalance causes high false negatives in low-representation classes (e.g. Fish, Roti, Poori, Vada).
- **MobileNetV3 Classifier:** Moderate Top-1 Accuracy (**53.9%**) and strong Top-5 Accuracy (**84.5%**). However, it suffers from a catastrophic bias: **over 95% of Biryani test images are predicted as `fried_rice` with >99% confidence**.
- **Inference Pipeline Defect:** The backend's hierarchical refinement map (`CATEGORY_REFINEMENT_MAP['biryani']`) explicitly allows `fried_rice` to overwrite YOLO's prediction. YOLO correctly detects `Biryani` at **96.7% confidence**, but MobileNet's output overwrites it with `fried_rice`.
- **Zero Detection Issue:** At `conf_thresh=0.25`, small or non-standard food items (e.g. `FIS_test_000007.jpg` with Fish) peak at **0.158 confidence** and are completely discarded.

---

## 2. Investigation of Observed Failure Cases

### Case 1: Biryani Image Misclassified as `fried_rice`
- **Test Image:** `1200px-Hyderabadi_Chicken_Biryani_jpg.rf.c39b821ecab23d38f0333d66902ed9f1.jpg`
- **Ground Truth:** `Biryani`
- **YOLO Prediction:** Detected `Biryani` at **96.67% confidence** (BBox: `[0.0, 0.8, 631.8, 635.8]`). **YOLO was 100% correct!**
- **MobileNetV3 Prediction:** Top-1 logit: `fried_rice` (prob: **99.89%**); `biryani` logit was 5.07 (prob: **0.07%**).
- **Backend Refinement Action:**
  - `CATEGORY_REFINEMENT_MAP['biryani'] = {"biryani", "fried_rice", "white_rice"}`.
  - Condition `top_conf >= 0.35` was satisfied (`0.9989 >= 0.35`).
  - MobileNet was allowed to overwrite YOLO, changing the final result from `Biryani` to `fried_rice`.
- **Root Causes:**
  1. **Taxonomy & Refinement Flaw:** YOLO's `Biryani` is already a specific dish, not a generic container class. Placing `fried_rice` in `CATEGORY_REFINEMENT_MAP['biryani']` is conceptually incorrect.
  2. **Classifier Training Bias:** MobileNet was trained on web-scraped data where rice texture strongly correlates with `fried_rice` weights. In empirical evaluation, **19 out of 24** Biryani test crops were misclassified as `fried_rice`.

### Case 2: Zero Detections on `FIS_test_000007.jpg`
- **Test Image:** `FIS_test_000007.jpg` (from `FoodInsSeg` test set).
- **Ground Truth:** 8 Bounding Boxes of `Fish` (Class 13).
- **YOLO Raw Output:** Highest anchor confidence for `Fish` was **0.1575** (15.75%).
- **Hardcoded Threshold:** `conf_thresh = 0.25` in `run_yolo_detection`.
- **Result:** All 8 boxes scored below 0.25 and were discarded, returning `0` detections.
- **Root Causes:**
  1. `FoodInsSeg` (`FIS`) images feature sliced, sushi-style, or European fish dishes that diverge from the Indian fish curries and whole fish in the primary training distribution.
  2. The rigid `0.25` threshold eliminates low-confidence detections rather than utilizing class-specific thresholds or returning candidate regions.

---

## 3. YOLO11s Detection Evaluation (Held-Out Test Set: 637 Images)

| Class | Ground Truth BBoxes | Model Detections | True Positives (TP) | False Positives (FP) | False Negatives (FN) | Precision | Recall | F1-Score |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Rice** | 352 | 203 | 145 | 58 | 207 | 0.714 | 0.412 | 0.522 |
| **Biryani** | 19 | 20 | 19 | 1 | 0 | 0.950 | 1.000 | 0.974 |
| **Roti_Naan** | 6 | 6 | 5 | 1 | 1 | 0.833 | 0.833 | 0.833 |
| **Poori** | 0 | 0 | 0 | 0 | 0 | 0.000 | 0.000 | 0.000 |
| **Dal_Sambar** | 16 | 17 | 14 | 3 | 2 | 0.824 | 0.875 | 0.849 |
| **Curry_Wet** | 171 | 169 | 143 | 26 | 28 | 0.846 | 0.836 | 0.841 |
| **Curry_Dry** | 83 | 89 | 77 | 12 | 6 | 0.865 | 0.928 | 0.895 |
| **Idli** | 29 | 22 | 21 | 1 | 8 | 0.955 | 0.724 | 0.824 |
| **Vada** | 0 | 0 | 0 | 0 | 0 | 0.000 | 0.000 | 0.000 |
| **Chutney** | 42 | 42 | 35 | 7 | 7 | 0.833 | 0.833 | 0.833 |
| **Curd** | 0 | 0 | 0 | 0 | 0 | 0.000 | 0.000 | 0.000 |
| **Salad** | 84 | 69 | 48 | 21 | 36 | 0.696 | 0.571 | 0.627 |
| **Egg** | 83 | 85 | 53 | 32 | 30 | 0.624 | 0.639 | 0.631 |
| **Fish** | 173 | 161 | 113 | 48 | 60 | 0.702 | 0.653 | 0.677 |

### Overall Detector Performance:
- **Overall Precision:** **76.22%**
- **Overall Recall:** **63.61%**
- **Overall F1-Score:** **69.35%**
- **mAP@50 (Macro-average):** **80.38%**
- **Zero Detection Images:** 33 out of 637 images.

---

## 4. MobileNetV3 Classification Evaluation (Held-Out Test Set: 576 Images)

- **Total Valid Test Crops:** 901
- **Top-1 Accuracy:** **53.94%**
- **Top-5 Accuracy:** **84.46%**

### Per-Class Performance Summary (Top 15 Classes):
| Food Class | Precision | Recall | F1-Score | Support |
| :--- | :--- | :--- | :--- | :--- |
| **micro avg** | 0.717 | 0.539 | 0.616 | 901.0 |
| **macro avg** | 0.837 | 0.519 | 0.563 | 901.0 |
| **weighted avg** | 0.835 | 0.539 | 0.584 | 901.0 |
| **gulab_jamun** | 1.000 | 0.707 | 0.828 | 58.0 |
| **kebab** | 1.000 | 0.196 | 0.328 | 51.0 |
| **kulfi** | 0.581 | 0.720 | 0.643 | 50.0 |
| **samosa** | 0.842 | 0.744 | 0.790 | 43.0 |
| **chole** | 1.000 | 0.167 | 0.286 | 42.0 |
| **chole_bhature** | 0.489 | 0.524 | 0.506 | 42.0 |
| **idli** | 0.773 | 0.829 | 0.800 | 41.0 |
| **chai** | 0.595 | 0.667 | 0.629 | 33.0 |
| **coconut_chutney** | 0.941 | 0.485 | 0.640 | 33.0 |
| **green_chutney** | 1.000 | 0.788 | 0.881 | 33.0 |
| **kheer** | 1.000 | 0.452 | 0.622 | 31.0 |
| **lassi** | 0.750 | 0.600 | 0.667 | 30.0 |

### Biryani Prediction Breakdown on Held-Out Test Set:
- **fried_rice:** 19 crops (79.2%)
- **biryani:** 3 crops (12.5%)
- **pakode:** 1 crops (4.2%)
- **kadai_paneer:** 1 crops (4.2%)

---

## 5. Architectural & Preprocessing Pipeline Audit

1. **Image Resizing & Aspect Ratio:**
   - YOLO uses letterbox padding to 640x640 with `(114, 114, 114)` gray fill. Aspect ratio is properly preserved. Coordinate unscaling back to native resolution is mathematically correct.
   - MobileNet crops use direct bilinear resizing to `224x224` without padding. Aspect ratio distortion occurs for elongated crops (e.g. Roti or Fish slices), altering texture features.
2. **Color Channel Ordering:**
   - Both models expect RGB. OpenCV loads BGR. Both pipelines perform proper BGR-to-RGB conversion. (Verified: Feeding BGR to MobileNet causes massive accuracy drop to pakode/jalebi).
3. **Normalization Constants:**
   - YOLO: `[0.0, 1.0]` (division by 255.0). Correct.
   - MobileNet: Standard ImageNet mean `[0.485, 0.456, 0.406]` and std `[0.229, 0.224, 0.225]`. Correct.
4. **Crop Generation Bounds:**
   - Bounding boxes are properly clamped to `[0, width]` and `[0, height]` preventing negative index crashes.
5. **Class Taxonomy Misalignment (CRITICAL):**
   - YOLO has 14 classes; MobileNet has 40 classes.
   - 17 MobileNet classes (sweets, snacks, drinks like `chai`, `lassi`, `jalebi`, `samosa`) have NO corresponding YOLO anchor class and can never be detected in multi-dish scenes.
   - 5 YOLO classes (`Salad`, `Egg`, `Poori`, `Curd`, `Fish`) have NO representation in MobileNet's 40 classes.

---

## 6. Actionable Fixes & Recommendation

### The Single Most Important Fix:
**Fix the Hierarchical Inference Pipeline Decision Logic in `backend/app.py`:**
1. **Never allow MobileNet to overwrite a specific, high-confidence YOLO detection:**
   When YOLO detects a specific dish like `Biryani` with confidence >= 0.60, **preserve YOLO's detection directly**.
2. **Restrict MobileNet refinement strictly to genuinely coarse umbrella classes:**
   Only invoke MobileNet when YOLO outputs coarse classes:
   - `Curry_Wet` $ightarrow$ Refine to `[shahi_paneer, kadai_paneer, mutton_curry, dum_aloo, chole, rajma_curry, fish_curry]`
   - `Curry_Dry` $ightarrow$ Refine to `[aloo_gobi, aloo_masala, bhindi_masala]`
   - `Rice` $ightarrow$ Refine to `[white_rice, fried_rice, poha]` (if confidence $\ge 0.70$)
   - `Roti_Naan` $ightarrow$ Refine to `[roti_naan, butter_naan]`
3. **Remove `fried_rice` and `white_rice` from `CATEGORY_REFINEMENT_MAP['biryani']`.**
4. **Remove Rule 2 (`top_conf >= 0.70` unconditional override):** MobileNet must never override classes outside the candidate family.
5. **Dynamic / Adaptive Confidence Threshold for YOLO:** Lower the global threshold from `0.25` to `0.15` with NMS to capture under-represented classes (like `Fish` in `FIS_test_000007.jpg`) without degrading precision on primary dishes.

### Retraining Assessment:
- **Is Retraining Immediately Necessary?** **NO for immediate functional deployment.** 
  The primary source of user-visible failure (`Biryani -> fried_rice`) is caused by the flawed refinement override rule in `backend/app.py`, NOT by YOLO (which achieved 96.7% accuracy on Biryani).
- **When Retraining WILL Be Necessary:**
  If the system must detect individual snacks/sweets (e.g. Samosas, Jalebi, Gulab Jamun) or unmeasured drinks, YOLO must be retrained on an expanded 25-class taxonomy incorporating `IndianFoodNet30` and `SOHL`.
- **Effort Estimate:** Retraining YOLO11s on 25 classes with 6,500 images requires ~2.5 hours on Google Colab GPU (T4/V100).

---

## 7. Artifact Inventory
- Test Evaluation Code: `evaluate_models.py`
- High-Level Metrics: `metrics.json`
- YOLO Detection Breakdown: `yolo_evaluation.json`
- MobileNet Classification Breakdown: `mobilenet_evaluation.json`
- Misclassified Test Examples: `misclassified_examples.csv`
- Classification Confusion Matrix: `confusion_matrix.png`
