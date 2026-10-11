"""
Experiment 2: Improve the Nutrition Database
============================================
Compares three nutrition database configurations:
  Config A: Current NUTRITION_DB (as-is baseline)
  Config B: Corrected & Source-Verified NUTRITION_DB (fixed egg USDA large, salad lipid baseline)
  Config C: Preparation-Specific NUTRITION_DB (differentiates preparations: scrambled vs boiled, dressed vs raw, lean vs oily fish)

Evaluates each configuration on:
  - Phase A: 54 reference items against ICMR-NIN IFCT 2017 & USDA FDC
  - Phase B: Weighed laboratory dish samples
"""

import os
import sys
import copy
import json
import csv
import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
EXP_DIR = os.path.join(BASE_DIR, "ml", "experiments", "experiment_nutrition_2")
RESULTS_DIR = os.path.join(EXP_DIR, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

sys.path.insert(0, os.path.join(BASE_DIR, "backend"))
from app import NUTRITION_DB, parse_standard_serving_weight

# -------------------------------------------------------------
# 1. DEFINE DATABASE CONFIGURATIONS
# -------------------------------------------------------------
# Config A: Current NUTRITION_DB
DB_CONFIG_A = copy.deepcopy(NUTRITION_DB)

# Config B: Corrected & Source-Verified NUTRITION_DB
DB_CONFIG_B = copy.deepcopy(NUTRITION_DB)
# Fix 1: Egg calibrated to USDA FDC ID 171287 (large hard-boiled egg 50g)
DB_CONFIG_B["egg"] = {"serving": "1 boiled (50g)", "cal": 78, "pro": 6.3, "carb": 0.6, "fat": 5.3}
# Fix 2: Salad calibrated to USDA FDC ID 170393 / IFCT 2017 (mixed raw vegetables 100g, 0.2g lipid)
DB_CONFIG_B["salad"] = {"serving": "1 portion (100g)", "cal": 21, "pro": 1.1, "carb": 4.2, "fat": 0.2}
# Fix 3: Fish fillet calibrated to USDA FDC ID 175176 / IFCT 2017 (cooked white fish fillet 100g)
DB_CONFIG_B["fish"] = {"serving": "1 piece (100g)", "cal": 142, "pro": 19.8, "carb": 0.0, "fat": 6.8}
# Fix 4: Lassi aligned serving weight to 205g (density 1.025 g/ml for 200ml)
DB_CONFIG_B["lassi"] = {"serving": "1 glass (205g)", "cal": 174, "pro": 6.4, "carb": 23.5, "fat": 5.8}

# Config C: Preparation-Specific NUTRITION_DB
# Adds dedicated culinary preparation profiles
DB_CONFIG_C = copy.deepcopy(DB_CONFIG_B)
PREPARATION_ENTRIES = {
    "scrambled_eggs": {"serving": "1 portion (100g)", "cal": 148, "pro": 10.0, "carb": 1.6, "fat": 11.0},
    "boiled_egg": {"serving": "1 piece (50g)", "cal": 78, "pro": 6.3, "carb": 0.6, "fat": 5.3},
    "caesar_salad": {"serving": "1 portion (100g)", "cal": 44, "pro": 3.2, "carb": 4.3, "fat": 2.1},
    "mixed_greens": {"serving": "1 portion (100g)", "cal": 28, "pro": 2.2, "carb": 5.3, "fat": 0.4},
    "steamed_white_fish": {"serving": "1 piece (100g)", "cal": 90, "pro": 18.4, "carb": 0.0, "fat": 1.5},
    "cooked_fish_fillet": {"serving": "1 piece (100g)", "cal": 142, "pro": 19.8, "carb": 0.0, "fat": 6.8}
}
DB_CONFIG_C.update(PREPARATION_ENTRIES)


# -------------------------------------------------------------
# 2. EVALUATION METRIC HELPER
# -------------------------------------------------------------
def compute_metrics(y_true, y_pred, nutrient_name=""):
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
        "n_valid_mape": num_valid,
        "mae": round(mae, 2),
        "rmse": round(rmse, 2),
        "mape_percent": round(mape, 2),
        "pct_within_10": round(within_10, 2),
        "pct_within_20": round(within_20, 2)
    }


# -------------------------------------------------------------
# 3. RUN DATABASE EXPERIMENT
# -------------------------------------------------------------
def run_experiment_2():
    print("=" * 70)
    print("EXPERIMENT 2: NUTRITION DATABASE IMPROVEMENT & COMPARISON")
    print("=" * 70)

    ref_csv = os.path.join(BASE_DIR, "ml", "evaluation", "nutrition", "reference_nutrition_ground_truth.csv")
    per_sample_csv = os.path.join(BASE_DIR, "ml", "evaluation", "nutrition", "per_sample_evaluation_results.csv")

    ref_df = pd.read_csv(ref_csv)
    sample_df = pd.read_csv(per_sample_csv)
    phase_b_df = sample_df[sample_df["phase"] == "Phase_B_Weighed_Samples"].copy()

    # --- PART 1: PHASE A EVALUATION ACROSS CONFIGS A, B, C ---
    configs = {
        "Config_A_Baseline": DB_CONFIG_A,
        "Config_B_Corrected": DB_CONFIG_B,
        "Config_C_Prep_Aware": DB_CONFIG_C
    }

    results = {}

    for cfg_name, db in configs.items():
        print(f"\nEvaluating Phase A on {cfg_name} (54 Reference Classes)...")
        y_t_cal, y_p_cal = [], []
        y_t_pro, y_p_pro = [], []
        y_t_carb, y_p_carb = [], []
        y_t_fat, y_p_fat = [], []

        for _, row in ref_df.iterrows():
            k = row["food_key"]
            entry = db[k]
            y_t_cal.append(float(row["ref_calories"]))
            y_p_cal.append(float(entry["cal"]))
            y_t_pro.append(float(row["ref_protein_g"]))
            y_p_pro.append(float(entry["pro"]))
            y_t_carb.append(float(row["ref_carbs_g"]))
            y_p_carb.append(float(entry["carb"]))
            y_t_fat.append(float(row["ref_fat_g"]))
            y_p_fat.append(float(entry["fat"]))

        results[f"Phase_A_{cfg_name}"] = {
            "calories": compute_metrics(y_t_cal, y_p_cal, "calories"),
            "protein": compute_metrics(y_t_pro, y_p_pro, "protein"),
            "carbs": compute_metrics(y_t_carb, y_p_carb, "carbs"),
            "fat": compute_metrics(y_t_fat, y_p_fat, "fat")
        }

    # --- PART 2: PHASE B EVALUATION ACROSS CONFIGS A, B, C ---
    for cfg_name, db in configs.items():
        print(f"Evaluating Phase B on {cfg_name} (160 Weighed Samples)...")
        y_t_cal, y_p_cal = [], []
        y_t_pro, y_p_pro = [], []
        y_t_carb, y_p_carb = [], []
        y_t_fat, y_p_fat = [], []

        for _, row in phase_b_df.iterrows():
            fkey = row["food_key"]
            dname = row["display_name"].lower()
            mass_g = float(row["mass_g"])

            # For Config C, use preparation-specific key if available
            lookup_key = fkey
            if cfg_name == "Config_C_Prep_Aware":
                if "scrambled" in dname and "scrambled_eggs" in db:
                    lookup_key = "scrambled_eggs"
                elif "caesar" in dname and "caesar_salad" in db:
                    lookup_key = "caesar_salad"
                elif "mixed greens" in dname and "mixed_greens" in db:
                    lookup_key = "mixed_greens"
                elif fkey == "fish" and "steamed_white_fish" in db:
                    lookup_key = "steamed_white_fish"

            entry = db[lookup_key]
            std_w = parse_standard_serving_weight(entry)
            scale = mass_g / std_w if std_w > 0 else 1.0

            p_cal = round(entry["cal"] * scale)
            p_pro = round(entry["pro"] * scale, 1)
            p_carb = round(entry["carb"] * scale, 1)
            p_fat = round(entry["fat"] * scale, 1)

            y_t_cal.append(float(row["ref_calories"]))
            y_p_cal.append(p_cal)
            y_t_pro.append(float(row["ref_protein_g"]))
            y_p_pro.append(p_pro)
            y_t_carb.append(float(row["ref_carbs_g"]))
            y_p_carb.append(p_carb)
            y_t_fat.append(float(row["ref_fat_g"]))
            y_p_fat.append(p_fat)

        results[f"Phase_B_{cfg_name}"] = {
            "calories": compute_metrics(y_t_cal, y_p_cal, "calories"),
            "protein": compute_metrics(y_t_pro, y_p_pro, "protein"),
            "carbs": compute_metrics(y_t_carb, y_p_carb, "carbs"),
            "fat": compute_metrics(y_t_fat, y_p_fat, "fat")
        }

    # Summary Table Output
    print("\n" + "=" * 95)
    print(f"{'Configuration':<25} | {'Cal MAE':<9} {'Cal MAPE':<10} | {'Pro MAE':<9} {'Pro MAPE':<10} | {'Fat MAE':<9} {'Fat MAPE':<10}")
    print("=" * 95)
    rows_summary = []
    for k, m in results.items():
        c_mae = m["calories"]["mae"]
        c_mape = m["calories"]["mape_percent"]
        p_mae = m["protein"]["mae"]
        p_mape = m["protein"]["mape_percent"]
        f_mae = m["fat"]["mae"]
        f_mape = m["fat"]["mape_percent"]
        print(f"{k:<25} | {c_mae:<9.2f} {c_mape:<9.2f}% | {p_mae:<9.2f} {p_mape:<9.2f}% | {f_mae:<9.2f} {f_mape:<9.2f}%")
        rows_summary.append({
            "configuration": k,
            "cal_mae": c_mae, "cal_rmse": m["calories"]["rmse"], "cal_mape": c_mape, "cal_pct10": m["calories"]["pct_within_10"], "cal_pct20": m["calories"]["pct_within_20"],
            "pro_mae": p_mae, "pro_rmse": m["protein"]["rmse"], "pro_mape": p_mape, "pro_pct10": m["protein"]["pct_within_10"], "pro_pct20": m["protein"]["pct_within_20"],
            "carb_mae": m["carbs"]["mae"], "carb_rmse": m["carbs"]["rmse"], "carb_mape": m["carbs"]["mape_percent"], "carb_pct10": m["carbs"]["pct_within_10"], "carb_pct20": m["carbs"]["pct_within_20"],
            "fat_mae": f_mae, "fat_rmse": m["fat"]["rmse"], "fat_mape": f_mape, "fat_pct10": m["fat"]["pct_within_10"], "fat_pct20": m["fat"]["pct_within_20"],
        })

    # Save to CSV and JSON
    out_csv = os.path.join(RESULTS_DIR, "exp2_database_comparison.csv")
    out_json = os.path.join(RESULTS_DIR, "exp2_database_comparison.json")

    pd.DataFrame(rows_summary).to_csv(out_csv, index=False)
    with open(out_json, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)

    print(f"\n[OK] Saved Experiment 2 comparison table to {out_csv}")
    print(f"[OK] Saved Experiment 2 metrics JSON to {out_json}")
    return results

if __name__ == "__main__":
    run_experiment_2()
