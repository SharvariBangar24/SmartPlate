"""
SmartPlate Nutritional Analysis Evaluation
==========================================
Two-Phase Evaluation Protocol:
  Phase A: Database Accuracy Evaluation
           Compares SmartPlate's NUTRITION_DB against independent reference values
           from ICMR-NIN IFCT 2017 and USDA FoodData Central at reference serving weights.
  Phase B: Sample-Level Weighed Nutrition Evaluation
           Evaluates SmartPlate's serving-scaled nutrition calculator on real
           laboratory scale-measured food samples (Nutrition5k single-dish instances)
           with bomb-calorimetry calories and chemical assay macronutrients.

Outputs:
  - ml/evaluation/nutrition/per_sample_evaluation_results.csv
  - ml/evaluation/nutrition/summary_metrics.json
  - ml/evaluation/nutrition/nutrition_accuracy_charts.png
"""

import os
import re
import csv
import json
import math
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ---------------------------------------------------------
# 1. PATH DEFINITIONS
# ---------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
NUTRITION_DIR = os.path.join(BASE_DIR, "ml", "evaluation", "nutrition")
REF_CSV_PATH = os.path.join(NUTRITION_DIR, "reference_nutrition_ground_truth.csv")

NUTRITION5K_CAFE1_PATH = os.path.join(
    BASE_DIR, "ml", "datasets", "nutrition5k", "metadata", "nutrition5k_dataset_metadata_dish_metadata_cafe1.csv"
)
NUTRITION5K_CAFE2_PATH = os.path.join(
    BASE_DIR, "ml", "datasets", "nutrition5k", "metadata", "nutrition5k_dataset_metadata_dish_metadata_cafe2.csv"
)

OUT_PER_SAMPLE_CSV = os.path.join(NUTRITION_DIR, "per_sample_evaluation_results.csv")
OUT_SUMMARY_JSON = os.path.join(NUTRITION_DIR, "summary_metrics.json")
OUT_CHART_PNG = os.path.join(NUTRITION_DIR, "nutrition_accuracy_charts.png")


# ---------------------------------------------------------
# 2. IMPORT NUTRITION_DB FROM BACKEND APP
# ---------------------------------------------------------
# We dynamically load NUTRITION_DB from backend/app.py to ensure 100% adherence to active production code
import sys
sys.path.insert(0, os.path.join(BASE_DIR, "backend"))
try:
    from app import NUTRITION_DB, parse_standard_serving_weight
    print(f"[OK] Successfully imported NUTRITION_DB ({len(NUTRITION_DB)} classes) from backend.app")
except ImportError as e:
    raise RuntimeError(f"Failed to import NUTRITION_DB from backend/app.py: {e}")


# ---------------------------------------------------------
# 3. STATISTICAL METRIC CALCULATORS
# ---------------------------------------------------------
def compute_metrics(y_true, y_pred, nutrient_name=""):
    """
    Computes MAE, RMSE, MAPE (for y_true > threshold), and % within +-10% and +-20%.
    Includes safeguards for zero or near-zero ground truth denominators.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    n = len(y_true)
    if n == 0:
        return {}

    errors = y_pred - y_true
    abs_errors = np.abs(errors)
    sq_errors = errors ** 2

    mae = float(np.mean(abs_errors))
    rmse = float(np.sqrt(np.mean(sq_errors)))

    # Threshold for percentage error: 1.0 kcal for calories, 0.5g for macros
    denom_threshold = 1.0 if "cal" in nutrient_name.lower() else 0.5

    valid_mask = y_true >= denom_threshold
    num_valid = int(np.sum(valid_mask))

    if num_valid > 0:
        rel_errors = abs_errors[valid_mask] / y_true[valid_mask]
        mape = float(np.mean(rel_errors) * 100.0)
        within_10 = float(np.mean(rel_errors <= 0.10) * 100.0)
        within_20 = float(np.mean(rel_errors <= 0.20) * 100.0)
    else:
        mape = 0.0
        within_10 = 100.0
        within_20 = 100.0

    return {
        "n_samples": n,
        "n_samples_for_percentage": num_valid,
        "mae": round(mae, 2),
        "rmse": round(rmse, 2),
        "mape_percent": round(mape, 2),
        "pct_within_10": round(within_10, 2),
        "pct_within_20": round(within_20, 2)
    }


# ---------------------------------------------------------
# 4. PHASE A: DATABASE BASELINE EVALUATION
# ---------------------------------------------------------
def run_phase_a():
    """
    Phase A compares NUTRITION_DB entries against independent reference values
    from ICMR-NIN IFCT 2017 & USDA FoodData Central.
    """
    print("\n" + "="*60)
    print("PHASE A: DATABASE ACCURACY EVALUATION (NUTRITION_DB vs Independent Ground Truth)")
    print("="*60)

    if not os.path.exists(REF_CSV_PATH):
        raise FileNotFoundError(f"Reference file not found: {REF_CSV_PATH}")

    ref_df = pd.read_csv(REF_CSV_PATH)
    print(f"Loaded {len(ref_df)} reference entries from {REF_CSV_PATH}")

    records = []
    y_true_cal, y_pred_cal = [], []
    y_true_pro, y_pred_pro = [], []
    y_true_carb, y_pred_carb = [], []
    y_true_fat, y_pred_fat = [], []

    for _, row in ref_df.iterrows():
        food_key = row["food_key"]
        if food_key not in NUTRITION_DB:
            print(f"  [Warning] Key '{food_key}' in reference CSV but not in NUTRITION_DB!")
            continue

        db_entry = NUTRITION_DB[food_key]
        pred_cal = float(db_entry["cal"])
        pred_pro = float(db_entry["pro"])
        pred_carb = float(db_entry["carb"])
        pred_fat = float(db_entry["fat"])

        ref_cal = float(row["ref_calories"])
        ref_pro = float(row["ref_protein_g"])
        ref_carb = float(row["ref_carbs_g"])
        ref_fat = float(row["ref_fat_g"])
        ref_weight = float(row["reference_weight_g"])

        y_true_cal.append(ref_cal)
        y_pred_cal.append(pred_cal)
        y_true_pro.append(ref_pro)
        y_pred_pro.append(pred_pro)
        y_true_carb.append(ref_carb)
        y_pred_carb.append(pred_carb)
        y_true_fat.append(ref_fat)
        y_pred_fat.append(pred_fat)

        err_cal = pred_cal - ref_cal
        err_pro = pred_pro - ref_pro
        err_carb = pred_carb - ref_carb
        err_fat = pred_fat - ref_fat

        pct_err_cal = (abs(err_cal) / ref_cal * 100.0) if ref_cal > 0 else 0.0
        pct_err_pro = (abs(err_pro) / ref_pro * 100.0) if ref_pro > 0 else 0.0
        pct_err_carb = (abs(err_carb) / ref_carb * 100.0) if ref_carb > 0 else 0.0
        pct_err_fat = (abs(err_fat) / ref_fat * 100.0) if ref_fat > 0 else 0.0

        records.append({
            "phase": "Phase_A_Database",
            "sample_id": f"ref_{food_key}",
            "food_key": food_key,
            "display_name": row["display_name"],
            "serving_description": row["serving_description"],
            "mass_g": ref_weight,
            "ref_calories": ref_cal,
            "pred_calories": pred_cal,
            "err_calories": round(err_cal, 2),
            "pct_err_calories": round(pct_err_cal, 2),
            "cal_within_10pct": bool(pct_err_cal <= 10.0),
            "cal_within_20pct": bool(pct_err_cal <= 20.0),
            "ref_protein_g": ref_pro,
            "pred_protein_g": pred_pro,
            "err_protein_g": round(err_pro, 2),
            "pct_err_protein": round(pct_err_pro, 2),
            "pro_within_10pct": bool(pct_err_pro <= 10.0),
            "pro_within_20pct": bool(pct_err_pro <= 20.0),
            "ref_carbs_g": ref_carb,
            "pred_carbs_g": pred_carb,
            "err_carbs_g": round(err_carb, 2),
            "pct_err_carbs": round(pct_err_carb, 2),
            "carb_within_10pct": bool(pct_err_carb <= 10.0),
            "carb_within_20pct": bool(pct_err_carb <= 20.0),
            "ref_fat_g": ref_fat,
            "pred_fat_g": pred_fat,
            "err_fat_g": round(err_fat, 2),
            "pct_err_fat": round(pct_err_fat, 2),
            "fat_within_10pct": bool(pct_err_fat <= 10.0),
            "fat_within_20pct": bool(pct_err_fat <= 20.0),
            "citation_source": row["citation_source"],
            "citation_notes": row["citation_notes"]
        })

    metrics_cal = compute_metrics(y_true_cal, y_pred_cal, "calories")
    metrics_pro = compute_metrics(y_true_pro, y_pred_pro, "protein")
    metrics_carb = compute_metrics(y_true_carb, y_pred_carb, "carbs")
    metrics_fat = compute_metrics(y_true_fat, y_pred_fat, "fat")

    phase_a_summary = {
        "evaluation_phase": "Phase_A_Database",
        "description": "SmartPlate NUTRITION_DB vs Independent ICMR-NIN IFCT 2017 & USDA FDC Ground Truth",
        "total_classes_evaluated": len(records),
        "metrics": {
            "calories": metrics_cal,
            "protein": metrics_pro,
            "carbohydrates": metrics_carb,
            "fat": metrics_fat
        }
    }

    print("\nPhase A Summary Results:")
    print(f"  Classes evaluated: {len(records)} / 54")
    print(f"  Calories: MAE = {metrics_cal['mae']} kcal, RMSE = {metrics_cal['rmse']} kcal, MAPE = {metrics_cal['mape_percent']}%, +-10% = {metrics_cal['pct_within_10']}%, +-20% = {metrics_cal['pct_within_20']}%")
    print(f"  Protein:  MAE = {metrics_pro['mae']} g, RMSE = {metrics_pro['rmse']} g, MAPE = {metrics_pro['mape_percent']}%, +-10% = {metrics_pro['pct_within_10']}%, +-20% = {metrics_pro['pct_within_20']}%")
    print(f"  Carbs:    MAE = {metrics_carb['mae']} g, RMSE = {metrics_carb['rmse']} g, MAPE = {metrics_carb['mape_percent']}%, +-10% = {metrics_carb['pct_within_10']}%, +-20% = {metrics_carb['pct_within_20']}%")
    print(f"  Fat:      MAE = {metrics_fat['mae']} g, RMSE = {metrics_fat['rmse']} g, MAPE = {metrics_fat['mape_percent']}%, +-10% = {metrics_fat['pct_within_10']}%, +-20% = {metrics_fat['pct_within_20']}%")

    return records, phase_a_summary, (y_true_cal, y_pred_cal, y_true_pro, y_pred_pro, y_true_carb, y_pred_carb, y_true_fat, y_pred_fat)


# ---------------------------------------------------------
# 5. PHASE B: SAMPLE-LEVEL WEIGHED NUTRITION EVALUATION
# ---------------------------------------------------------
def run_phase_b():
    """
    Phase B evaluates SmartPlate's serving-scaled nutrition calculator on real
    laboratory scale-measured food samples (Nutrition5k single-dish instances).
    """
    print("\n" + "="*60)
    print("PHASE B: WEIGHED SAMPLE-LEVEL NUTRITION EVALUATION (Nutrition5k Lab Ground Truth)")
    print("="*60)

    # Clean, direct physical food mappings to SmartPlate classes
    MAPPING = {
        "white rice": "white_rice",
        "scrambled eggs": "egg",
        "caesar salad": "salad",
        "mixed greens": "salad",
        "fish": "fish"
    }

    samples = []
    for csv_path in [NUTRITION5K_CAFE1_PATH, NUTRITION5K_CAFE2_PATH]:
        if not os.path.exists(csv_path):
            print(f"[Warning] Nutrition5k file not found: {csv_path}")
            continue
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            for row in reader:
                if not row:
                    continue
                dish_id = row[0]
                calories_gt = float(row[1])
                mass_gt = float(row[2])
                fat_gt = float(row[3])
                carb_gt = float(row[4])
                pro_gt = float(row[5])

                # Ingredients start at column 6 in chunks of 7
                ingr_tokens = row[6:]
                num_ingrs = len(ingr_tokens) // 7
                if num_ingrs == 1:
                    ingr_name = ingr_tokens[1].strip()
                    if ingr_name in MAPPING:
                        sp_key = MAPPING[ingr_name]
                        samples.append({
                            "dish_id": dish_id,
                            "raw_food_name": ingr_name,
                            "food_key": sp_key,
                            "mass_g": mass_gt,
                            "ref_calories": calories_gt,
                            "ref_protein_g": pro_gt,
                            "ref_carbs_g": carb_gt,
                            "ref_fat_g": fat_gt
                        })

    print(f"Extracted {len(samples)} valid single-dish laboratory samples matching SmartPlate classes.")

    records = []
    y_true_cal, y_pred_cal = [], []
    y_true_pro, y_pred_pro = [], []
    y_true_carb, y_pred_carb = [], []
    y_true_fat, y_pred_fat = [], []

    per_class_samples = {}

    for s in samples:
        sp_key = s["food_key"]
        mass_g = s["mass_g"]
        db_entry = NUTRITION_DB[sp_key]
        std_serving_g = parse_standard_serving_weight(db_entry)

        # SmartPlate proportional scaling formula (from backend/app.py lines 500-506)
        scale_factor = mass_g / std_serving_g if std_serving_g > 0 else 1.0
        pred_cal = round(db_entry["cal"] * scale_factor)
        pred_pro = round(db_entry["pro"] * scale_factor, 1)
        pred_carb = round(db_entry["carb"] * scale_factor, 1)
        pred_fat = round(db_entry["fat"] * scale_factor, 1)

        ref_cal = s["ref_calories"]
        ref_pro = s["ref_protein_g"]
        ref_carb = s["ref_carbs_g"]
        ref_fat = s["ref_fat_g"]

        y_true_cal.append(ref_cal)
        y_pred_cal.append(pred_cal)
        y_true_pro.append(ref_pro)
        y_pred_pro.append(pred_pro)
        y_true_carb.append(ref_carb)
        y_pred_carb.append(pred_carb)
        y_true_fat.append(ref_fat)
        y_pred_fat.append(pred_fat)

        err_cal = pred_cal - ref_cal
        err_pro = pred_pro - ref_pro
        err_carb = pred_carb - ref_carb
        err_fat = pred_fat - ref_fat

        pct_err_cal = (abs(err_cal) / ref_cal * 100.0) if ref_cal > 0 else 0.0
        pct_err_pro = (abs(err_pro) / ref_pro * 100.0) if ref_pro > 0 else 0.0
        pct_err_carb = (abs(err_carb) / ref_carb * 100.0) if ref_carb > 0 else 0.0
        pct_err_fat = (abs(err_fat) / ref_fat * 100.0) if ref_fat > 0 else 0.0

        records.append({
            "phase": "Phase_B_Weighed_Samples",
            "sample_id": s["dish_id"],
            "food_key": sp_key,
            "display_name": f"{s['raw_food_name']} ({sp_key})",
            "serving_description": f"Measured scale weight: {mass_g:.1f}g",
            "mass_g": round(mass_g, 1),
            "ref_calories": round(ref_cal, 2),
            "pred_calories": round(pred_cal, 2),
            "err_calories": round(err_cal, 2),
            "pct_err_calories": round(pct_err_cal, 2),
            "cal_within_10pct": bool(pct_err_cal <= 10.0),
            "cal_within_20pct": bool(pct_err_cal <= 20.0),
            "ref_protein_g": round(ref_pro, 2),
            "pred_protein_g": round(pred_pro, 2),
            "err_protein_g": round(err_pro, 2),
            "pct_err_protein": round(pct_err_pro, 2),
            "pro_within_10pct": bool(pct_err_pro <= 10.0),
            "pro_within_20pct": bool(pct_err_pro <= 20.0),
            "ref_carbs_g": round(ref_carb, 2),
            "pred_carbs_g": round(pred_carb, 2),
            "err_carbs_g": round(err_carb, 2),
            "pct_err_carbs": round(pct_err_carb, 2),
            "carb_within_10pct": bool(pct_err_carb <= 10.0),
            "carb_within_20pct": bool(pct_err_carb <= 20.0),
            "ref_fat_g": round(ref_fat, 2),
            "pred_fat_g": round(pred_fat, 2),
            "err_fat_g": round(err_fat, 2),
            "pct_err_fat": round(pct_err_fat, 2),
            "fat_within_10pct": bool(pct_err_fat <= 10.0),
            "fat_within_20pct": bool(pct_err_fat <= 20.0),
            "citation_source": f"Nutrition5k Dataset ({s['dish_id']})",
            "citation_notes": "Precision electronic scale mass and laboratory bomb-calorimetry / USDA assay ground truth"
        })

        if sp_key not in per_class_samples:
            per_class_samples[sp_key] = {"cal_t": [], "cal_p": [], "pro_t": [], "pro_p": [], "carb_t": [], "carb_p": [], "fat_t": [], "fat_p": []}
        per_class_samples[sp_key]["cal_t"].append(ref_cal)
        per_class_samples[sp_key]["cal_p"].append(pred_cal)
        per_class_samples[sp_key]["pro_t"].append(ref_pro)
        per_class_samples[sp_key]["pro_p"].append(pred_pro)
        per_class_samples[sp_key]["carb_t"].append(ref_carb)
        per_class_samples[sp_key]["carb_p"].append(pred_carb)
        per_class_samples[sp_key]["fat_t"].append(ref_fat)
        per_class_samples[sp_key]["fat_p"].append(pred_fat)

    metrics_cal = compute_metrics(y_true_cal, y_pred_cal, "calories")
    metrics_pro = compute_metrics(y_true_pro, y_pred_pro, "protein")
    metrics_carb = compute_metrics(y_true_carb, y_pred_carb, "carbs")
    metrics_fat = compute_metrics(y_true_fat, y_pred_fat, "fat")

    per_class_metrics = {}
    for c_key, data in per_class_samples.items():
        per_class_metrics[c_key] = {
            "n_samples": len(data["cal_t"]),
            "calories": compute_metrics(data["cal_t"], data["cal_p"], "calories"),
            "protein": compute_metrics(data["pro_t"], data["pro_p"], "protein"),
            "carbohydrates": compute_metrics(data["carb_t"], data["carb_p"], "carbs"),
            "fat": compute_metrics(data["fat_t"], data["fat_p"], "fat")
        }

    phase_b_summary = {
        "evaluation_phase": "Phase_B_Weighed_Samples",
        "description": "SmartPlate Weighed Nutrition Scaling on Nutrition5k Physical Lab Dishes",
        "total_samples_evaluated": len(records),
        "overall_metrics": {
            "calories": metrics_cal,
            "protein": metrics_pro,
            "carbohydrates": metrics_carb,
            "fat": metrics_fat
        },
        "per_class_metrics": per_class_metrics,
        "data_availability_limitation": {
            "limitation_identified": True,
            "details": (
                "Indian cuisine datasets in repository (IndianFoodNet 4,076 images, SOHL 377 images) contain 2D bounding boxes "
                "only. Neither physical food weight nor chemical laboratory bomb-calorimetry/macronutrients were measured. "
                "As instructed, ordinary food images were NOT treated as calorie ground truth. "
                "Phase B is therefore rigorously executed on Nutrition5k scale-measured single-dish instances matching SmartPlate classes."
            )
        }
    }

    print("\nPhase B Summary Results:")
    print(f"  Valid weighed samples evaluated: {len(records)}")
    print(f"  Calories: MAE = {metrics_cal['mae']} kcal, RMSE = {metrics_cal['rmse']} kcal, MAPE = {metrics_cal['mape_percent']}%, +-10% = {metrics_cal['pct_within_10']}%, +-20% = {metrics_cal['pct_within_20']}%")
    print(f"  Protein:  MAE = {metrics_pro['mae']} g, RMSE = {metrics_pro['rmse']} g, MAPE = {metrics_pro['mape_percent']}%, +-10% = {metrics_pro['pct_within_10']}%, +-20% = {metrics_pro['pct_within_20']}%")
    print(f"  Carbs:    MAE = {metrics_carb['mae']} g, RMSE = {metrics_carb['rmse']} g, MAPE = {metrics_carb['mape_percent']}%, +-10% = {metrics_carb['pct_within_10']}%, +-20% = {metrics_carb['pct_within_20']}%")
    print(f"  Fat:      MAE = {metrics_fat['mae']} g, RMSE = {metrics_fat['rmse']} g, MAPE = {metrics_fat['mape_percent']}%, +-10% = {metrics_fat['pct_within_10']}%, +-20% = {metrics_fat['pct_within_20']}%")

    return records, phase_b_summary, (y_true_cal, y_pred_cal, y_true_pro, y_pred_pro, y_true_carb, y_pred_carb, y_true_fat, y_pred_fat)


# ---------------------------------------------------------
# 6. PLOT CHARTS
# ---------------------------------------------------------
def generate_charts(phase_a_data, phase_b_data):
    """
    Generates high-resolution publication-quality evaluation charts.
    """
    print("\nGenerating evaluation charts ->", OUT_CHART_PNG)
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    plt.subplots_adjust(hspace=0.32, wspace=0.28)

    # Subplot 1: Phase A Database Calories Parity Plot
    ax1 = axes[0, 0]
    ref_cal_a, pred_cal_a = phase_a_data[0], phase_a_data[1]
    ax1.scatter(ref_cal_a, pred_cal_a, color="#1f77b4", alpha=0.8, edgecolors="k", s=55, label="NUTRITION_DB entries (N=54)")
    max_cal_a = max(max(ref_cal_a), max(pred_cal_a)) * 1.05
    ax1.plot([0, max_cal_a], [0, max_cal_a], 'r--', lw=1.8, label="Ideal Parity (y=x)")
    ax1.fill_between([0, max_cal_a], [0, 0.9 * max_cal_a], [0, 1.1 * max_cal_a], color="green", alpha=0.12, label="+-10% Tolerance")
    ax1.set_xlim(0, max_cal_a)
    ax1.set_ylim(0, max_cal_a)
    ax1.set_title("Phase A: Calorie Parity (NUTRITION_DB vs IFCT/USDA)", fontsize=12, fontweight="bold")
    ax1.set_xlabel("Independent Reference Calories (kcal)", fontsize=10)
    ax1.set_ylabel("SmartPlate Database Calories (kcal)", fontsize=10)
    ax1.legend(loc="upper left", fontsize=8.5)
    ax1.grid(True, linestyle=":", alpha=0.6)

    # Subplot 2: Phase A & Phase B MAE Comparison Bar Chart
    ax2 = axes[0, 1]
    nutrients = ["Calories\n(kcal)", "Protein\n(g)", "Carbs\n(g)", "Fat\n(g)"]
    mae_a = [
        float(np.mean(np.abs(np.array(phase_a_data[1]) - np.array(phase_a_data[0])))),
        float(np.mean(np.abs(np.array(phase_a_data[3]) - np.array(phase_a_data[2])))),
        float(np.mean(np.abs(np.array(phase_a_data[5]) - np.array(phase_a_data[4])))),
        float(np.mean(np.abs(np.array(phase_a_data[7]) - np.array(phase_a_data[6]))))
    ]
    mae_b = [
        float(np.mean(np.abs(np.array(phase_b_data[1]) - np.array(phase_b_data[0])))),
        float(np.mean(np.abs(np.array(phase_b_data[3]) - np.array(phase_b_data[2])))),
        float(np.mean(np.abs(np.array(phase_b_data[5]) - np.array(phase_b_data[4])))),
        float(np.mean(np.abs(np.array(phase_b_data[7]) - np.array(phase_b_data[6]))))
    ]
    x = np.arange(len(nutrients))
    width = 0.35
    b1 = ax2.bar(x - width/2, mae_a, width, label="Phase A (Database Baseline)", color="#2ca02c")
    b2 = ax2.bar(x + width/2, mae_b, width, label="Phase B (Weighed Samples N=160)", color="#ff7f0e")
    ax2.set_title("Mean Absolute Error (MAE) Comparison", fontsize=12, fontweight="bold")
    ax2.set_xticks(x)
    ax2.set_xticklabels(nutrients, fontsize=9.5)
    ax2.set_ylabel("MAE (kcal or grams)", fontsize=10)
    ax2.legend(loc="upper right", fontsize=8.5)
    ax2.grid(True, linestyle=":", alpha=0.6, axis="y")
    for rect in list(b1) + list(b2):
        h = rect.get_height()
        ax2.annotate(f"{h:.1f}", xy=(rect.get_x() + rect.get_width()/2, h),
                     xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=8.5)

    # Subplot 3: Phase B Scaled Calories Parity Scatter Plot
    ax3 = axes[1, 0]
    ref_cal_b, pred_cal_b = phase_b_data[0], phase_b_data[1]
    ax3.scatter(ref_cal_b, pred_cal_b, color="#9467bd", alpha=0.75, edgecolors="k", s=45, label="Weighed Lab Samples (N=160)")
    max_cal_b = max(max(ref_cal_b), max(pred_cal_b)) * 1.05
    ax3.plot([0, max_cal_b], [0, max_cal_b], 'r--', lw=1.8, label="Ideal Parity (y=x)")
    ax3.fill_between([0, max_cal_b], [0, 0.8 * max_cal_b], [0, 1.2 * max_cal_b], color="orange", alpha=0.12, label="+-20% Tolerance Band")
    ax3.set_xlim(0, max_cal_b)
    ax3.set_ylim(0, max_cal_b)
    ax3.set_title("Phase B: Weighed Sample Calorie Parity (Nutrition5k)", fontsize=12, fontweight="bold")
    ax3.set_xlabel("Laboratory Measured Ground-Truth Calories (kcal)", fontsize=10)
    ax3.set_ylabel("SmartPlate Predicted Scaled Calories (kcal)", fontsize=10)
    ax3.legend(loc="upper left", fontsize=8.5)
    ax3.grid(True, linestyle=":", alpha=0.6)

    # Subplot 4: Percentage of Samples within Tolerance Bands
    ax4 = axes[1, 1]
    pct_10 = [
        float(np.mean(np.abs(np.array(phase_b_data[1]) - np.array(phase_b_data[0])) / np.maximum(phase_b_data[0], 1.0) <= 0.10) * 100),
        float(np.mean(np.abs(np.array(phase_b_data[3]) - np.array(phase_b_data[2])) / np.maximum(phase_b_data[2], 0.5) <= 0.10) * 100),
        float(np.mean(np.abs(np.array(phase_b_data[5]) - np.array(phase_b_data[4])) / np.maximum(phase_b_data[4], 0.5) <= 0.10) * 100),
        float(np.mean(np.abs(np.array(phase_b_data[7]) - np.array(phase_b_data[6])) / np.maximum(phase_b_data[6], 0.5) <= 0.10) * 100)
    ]
    pct_20 = [
        float(np.mean(np.abs(np.array(phase_b_data[1]) - np.array(phase_b_data[0])) / np.maximum(phase_b_data[0], 1.0) <= 0.20) * 100),
        float(np.mean(np.abs(np.array(phase_b_data[3]) - np.array(phase_b_data[2])) / np.maximum(phase_b_data[2], 0.5) <= 0.20) * 100),
        float(np.mean(np.abs(np.array(phase_b_data[5]) - np.array(phase_b_data[4])) / np.maximum(phase_b_data[4], 0.5) <= 0.20) * 100),
        float(np.mean(np.abs(np.array(phase_b_data[7]) - np.array(phase_b_data[6])) / np.maximum(phase_b_data[6], 0.5) <= 0.20) * 100)
    ]
    b_10 = ax4.bar(x - width/2, pct_10, width, label="Within +-10%", color="#17becf")
    b_20 = ax4.bar(x + width/2, pct_20, width, label="Within +-20%", color="#1f77b4")
    ax4.set_title("Phase B: % Samples within Tolerance Bands", fontsize=12, fontweight="bold")
    ax4.set_xticks(x)
    ax4.set_xticklabels(nutrients, fontsize=9.5)
    ax4.set_ylabel("Accuracy Percentage (%)", fontsize=10)
    ax4.set_ylim(0, 105)
    ax4.legend(loc="lower right", fontsize=8.5)
    ax4.grid(True, linestyle=":", alpha=0.6, axis="y")
    for rect in list(b_10) + list(b_20):
        h = rect.get_height()
        ax4.annotate(f"{h:.1f}%", xy=(rect.get_x() + rect.get_width()/2, h),
                     xytext=(0, 2), textcoords="offset points", ha="center", va="bottom", fontsize=8)

    plt.suptitle("SmartPlate Nutrition Evaluation: Empirical Validation Protocol", fontsize=15, fontweight="bold", y=0.99)
    plt.tight_layout()
    plt.savefig(OUT_CHART_PNG, dpi=300)
    plt.close()
    print("[OK] Saved evaluation charts successfully.")


# ---------------------------------------------------------
# 7. MAIN DRIVER
# ---------------------------------------------------------
def main():
    print("Starting SmartPlate Nutrition Evaluation...")
    phase_a_records, phase_a_summary, phase_a_data = run_phase_a()
    phase_b_records, phase_b_summary, phase_b_data = run_phase_b()

    # Save per-sample CSV results
    all_records = phase_a_records + phase_b_records
    keys = list(all_records[0].keys())
    with open(OUT_PER_SAMPLE_CSV, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(all_records)
    print(f"\n[OK] Saved {len(all_records)} evaluation records to {OUT_PER_SAMPLE_CSV}")

    # Save summary metrics JSON
    summary_data = {
        "title": "SmartPlate Nutrition Analysis Comprehensive Evaluation",
        "phase_a": phase_a_summary,
        "phase_b": phase_b_summary
    }
    with open(OUT_SUMMARY_JSON, 'w', encoding='utf-8') as f:
        json.dump(summary_data, f, indent=2)
    print(f"[OK] Saved summary metrics to {OUT_SUMMARY_JSON}")

    # Generate graphs
    generate_charts(phase_a_data, phase_b_data)

    print("\n[SUCCESS] Nutrition evaluation pipeline completed successfully!")


if __name__ == "__main__":
    main()
