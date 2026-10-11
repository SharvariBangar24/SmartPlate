"""
Experiment 7: Full Pipeline Validation & Propagation Analysis
=============================================================
Analyzes end-to-end uncertainty and error propagation across the 4-stage pipeline:
  Stage 1: YOLO11s Object Detection (Localization & Coarse Classification)
  Stage 2: MobileNetV3 Hierarchical Fine-Grained Refinement
  Stage 3: 2D Geometric Portion Estimation Regressor
  Stage 4: Scaled Nutrition Calculator

Quantifies:
  - Error propagation formulas and compounding sensitivity
  - Calorie and macro uncertainty intervals for top Indian dishes given portion MAE (61.61g)
  - End-to-end execution on available test images
  - Formal disclosure of data limits preventing global end-to-end calorie benchmark on Indian dishes
"""

import os
import sys
import json
import cv2
import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
EXP_DIR = os.path.join(BASE_DIR, "ml", "experiments", "experiment_nutrition_2")
RESULTS_DIR = os.path.join(EXP_DIR, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

sys.path.insert(0, os.path.join(BASE_DIR, "backend"))
from app import (
    NUTRITION_DB, parse_standard_serving_weight,
    run_yolo_detection, classify_crop, estimate_food_portion
)

def run_experiment_7():
    print("=" * 70)
    print("EXPERIMENT 7: FULL PIPELINE ERROR PROPAGATION ANALYSIS")
    print("=" * 70)

    # 1. Error Propagation Analysis
    # Given portion weight error delta_M = 61.61g (HistGradientBoosting test MAE)
    # The calorie error induced is delta_Cal = delta_M * (Cal_db / Serving_g)
    PORTION_MAE_G = 61.61

    dishes_to_analyze = [
        ("biryani", 300, 450, 15.0, 55.0, 18.0),
        ("white_rice", 158, 205, 4.3, 44.5, 0.4),
        ("dal_sambar", 150, 130, 7.0, 20.0, 3.0),
        ("curry_wet", 150, 180, 6.0, 15.0, 10.0),
        ("kadai_paneer", 150, 260, 12.0, 10.0, 19.0),
        ("roti_naan", 40, 120, 3.5, 20.0, 2.0),
        ("chole_bhature", 250, 480, 12.0, 62.0, 20.0),
        ("egg", 50, 78, 6.3, 0.6, 5.3),
        ("salad", 100, 21, 1.1, 4.2, 0.2)
    ]

    propagation_results = []
    for name, s_g, cal, pro, carb, fat in dishes_to_analyze:
        cal_density = cal / s_g
        pro_density = pro / s_g
        carb_density = carb / s_g
        fat_density = fat / s_g

        induced_cal_err = PORTION_MAE_G * cal_density
        induced_pro_err = PORTION_MAE_G * pro_density
        induced_carb_err = PORTION_MAE_G * carb_density
        induced_fat_err = PORTION_MAE_G * fat_density

        propagation_results.append({
            "dish_name": name,
            "standard_serving_g": s_g,
            "cal_density_kcal_per_g": round(cal_density, 3),
            "portion_mae_g": PORTION_MAE_G,
            "induced_calorie_mae_kcal": round(induced_cal_err, 2),
            "induced_calorie_error_pct_of_serving": round((induced_cal_err / cal) * 100.0, 2),
            "induced_protein_mae_g": round(induced_pro_err, 2),
            "induced_carbs_mae_g": round(induced_carb_err, 2),
            "induced_fat_mae_g": round(induced_fat_err, 2)
        })

    prop_df = pd.DataFrame(propagation_results)
    out_csv = os.path.join(RESULTS_DIR, "exp7_pipeline_propagation.csv")
    out_json = os.path.join(RESULTS_DIR, "exp7_pipeline_propagation.json")
    prop_df.to_csv(out_csv, index=False)

    print("\nEnd-to-End Error Compounding from Portion Estimation (MAE = 61.61g):")
    for _, r in prop_df.iterrows():
        print(f"  {r['dish_name']:15s}: Induced Calorie Error = {r['induced_calorie_mae_kcal']:6.1f} kcal ({r['induced_calorie_error_pct_of_serving']:5.1f}% of standard serving)")

    # 2. Test Image Execution Audit (Real Uploaded Image Pipeline)
    test_img_path = os.path.join(BASE_DIR, "frontend", "test_biryani.jpg")
    img_exec_results = {}
    if os.path.exists(test_img_path):
        img = cv2.imread(test_img_path)
        h, w = img.shape[:2]
        dets = run_yolo_detection(img)
        print(f"\n[Real Image Pipeline Test] Tested '{os.path.basename(test_img_path)}' ({w}x{h})")
        print(f"  Detections: {len(dets)} food regions found.")
        for i, d in enumerate(dets):
            x1, y1, x2, y2 = d["bbox"]
            crop = img[int(y1):int(y2), int(x1):int(x2)]
            fname, fconf = classify_crop(crop, d["yolo_name"], d["confidence"])
            db_k = fname.lower().replace(" ", "_")
            nutri = NUTRITION_DB.get(db_k, NUTRITION_DB["unknown"])
            std_w = parse_standard_serving_weight(nutri)
            est_w, method, is_m = estimate_food_portion(d["bbox"], w, h, fname, std_w)
            scale = est_w / std_w if std_w > 0 else 1.0
            p_cal = round(nutri["cal"] * scale)
            print(f"  Food {i+1}: {fname} (conf: {fconf:.2f}) -> Est Weight: {est_w}g ({method}) -> Calories: {p_cal} kcal")
            img_exec_results[f"food_{i+1}"] = {
                "name": fname, "confidence": round(float(fconf), 3),
                "estimated_weight_g": est_w, "method": method,
                "scaled_calories": p_cal
            }

    final_payload = {
        "portion_regressor_test_mae_g": PORTION_MAE_G,
        "error_propagation_by_dish": propagation_results,
        "real_image_test": img_exec_results,
        "scientific_conclusion": (
            "Because Indian food datasets in literature (IndianFoodNet, SOHL) contain bounding boxes only and lack "
            "chemical bomb calorimetry, global end-to-end calorie accuracy cannot be stated without calibrated reference markers. "
            "The error is predominantly governed by the 2D geometric portion regressor (MAE = 61.61g), which scales proportionally "
            "with the dish's caloric density (ranging from 12.9 kcal for raw salad to 184.8 kcal for dense roti)."
        )
    }

    with open(out_json, 'w', encoding='utf-8') as f:
        json.dump(final_payload, f, indent=2)

    print(f"\n[OK] Saved Experiment 7 propagation analysis to {out_csv} and {out_json}")
    return final_payload

if __name__ == "__main__":
    run_experiment_7()
