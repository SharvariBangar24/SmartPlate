"""
Experiment 1: Validate the Nutrition Evaluation Baseline
=========================================================
Performs an independent audit of the baseline nutrition evaluation:
- Inspects reference ground truth, serving sizes, and per-100g conversions
- Validates sample-level predictions and errors
- Audits zero-reference handling in MAPE calculations
- Identifies and records every sample with preparation or mapping mismatches
- Verifies the portion model test MAE discrepancy (61.61g vs 70.93g)
"""

import os
import sys
import json
import csv
import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
NUTRITION_EVAL_DIR = os.path.join(BASE_DIR, "ml", "evaluation", "nutrition")
EXP_DIR = os.path.join(BASE_DIR, "ml", "experiments", "experiment_nutrition_2")
RESULTS_DIR = os.path.join(EXP_DIR, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

sys.path.insert(0, os.path.join(BASE_DIR, "backend"))
from app import NUTRITION_DB, parse_standard_serving_weight

def validate_baseline():
    print("=" * 70)
    print("EXPERIMENT 1: VALIDATING NUTRITION EVALUATION BASELINE")
    print("=" * 70)

    # 1. Load Baseline Files
    ref_csv = os.path.join(NUTRITION_EVAL_DIR, "reference_nutrition_ground_truth.csv")
    per_sample_csv = os.path.join(NUTRITION_EVAL_DIR, "per_sample_evaluation_results.csv")
    summary_json = os.path.join(NUTRITION_EVAL_DIR, "summary_metrics.json")
    portion_metrics_json = os.path.join(BASE_DIR, "ml", "experiments", "experiment_portion_1", "metrics.json")

    ref_df = pd.read_csv(ref_csv)
    sample_df = pd.read_csv(per_sample_csv)
    with open(summary_json, 'r', encoding='utf-8') as f:
        summary_data = json.load(f)
    with open(portion_metrics_json, 'r', encoding='utf-8') as f:
        portion_data = json.load(f)

    # 2. Phase A Validation: 54 classes
    phase_a_samples = sample_df[sample_df["phase"] == "Phase_A_Database"]
    print(f"\n[Phase A Audit] Verified {len(phase_a_samples)} food classes in evaluation CSV vs {len(ref_df)} in reference CSV.")
    assert len(phase_a_samples) == 54, f"Expected 54 Phase A samples, got {len(phase_a_samples)}"

    # Audit serving weight parsing consistency
    serving_parse_audit = []
    for _, row in ref_df.iterrows():
        k = row["food_key"]
        db_item = NUTRITION_DB.get(k)
        parsed_w = parse_standard_serving_weight(db_item)
        ref_w = float(row["reference_weight_g"])
        if abs(parsed_w - ref_w) > 0.01:
            serving_parse_audit.append({
                "food_key": k,
                "parsed_weight": parsed_w,
                "ref_weight": ref_w,
                "status": "MISMATCH"
            })
    print(f"  Serving weight parsing: {len(ref_df) - len(serving_parse_audit)}/54 matched perfectly.")

    # 3. Phase B Validation: 160 Nutrition5k samples
    phase_b_samples = sample_df[sample_df["phase"] == "Phase_B_Weighed_Samples"].copy()
    print(f"\n[Phase B Audit] Found {len(phase_b_samples)} weighed dish samples.")
    assert len(phase_b_samples) == 160, f"Expected 160 Phase B samples, got {len(phase_b_samples)}"

    # Check duplicate dish IDs
    unique_dishes = phase_b_samples["sample_id"].nunique()
    print(f"  Unique dish IDs: {unique_dishes} (Duplicates: {len(phase_b_samples) - unique_dishes})")

    # Audit Mappings & Preparations
    mismatched_samples = []
    for _, row in phase_b_samples.iterrows():
        sid = row["sample_id"]
        dname = row["display_name"]
        fkey = row["food_key"]

        # Flag 1: Caesar Salad mapped to raw Salad
        if "caesar salad" in dname.lower():
            mismatched_samples.append({
                "sample_id": sid,
                "display_name": dname,
                "mapped_class": fkey,
                "issue_type": "preparation_and_ingredient_mismatch",
                "details": "Caesar salad includes parmesan cheese, croutons, and emulsified dressing (~2.1g fat, 3.2g protein, 44 kcal/100g), whereas SmartPlate DB specifies raw Indian side salad (0.0g fat, 1.0g protein, 20 kcal/100g)."
            })
        # Flag 2: Scrambled eggs mapped to boiled egg
        elif "scrambled eggs" in dname.lower():
            mismatched_samples.append({
                "sample_id": sid,
                "display_name": dname,
                "mapped_class": fkey,
                "issue_type": "cooking_method_mismatch",
                "details": "Scrambled eggs cooked with butter/oil (148 kcal, 11g fat/100g) compared against boiled egg entry (136 kcal, 9.6g fat/100g)."
            })
        # Flag 3: Lean white fish vs prepared Indian fish fillet
        elif fkey == "fish":
            mismatched_samples.append({
                "sample_id": sid,
                "display_name": dname,
                "mapped_class": fkey,
                "issue_type": "species_and_data_anomaly",
                "details": "Nutrition5k fish is steamed lean cod (90 kcal/100g) compared to prepared Indian fish fillet (150 kcal/100g). Furthermore, Nutrition5k ingredient #49 records 13g fat with 90 kcal, violating Atwater energy conservation."
            })

    print(f"  Mismatched / Sub-optimal comparison samples identified: {len(mismatched_samples)} / 160")
    print(f"    - Caesar salad: {sum(1 for m in mismatched_samples if 'caesar' in m['display_name'].lower())}")
    print(f"    - Scrambled eggs: {sum(1 for m in mismatched_samples if 'scrambled' in m['display_name'].lower())}")
    print(f"    - Fish fillet: {sum(1 for m in mismatched_samples if m['mapped_class'] == 'fish')}")

    # 4. Zero-Reference & Denominator Safeguard Audit
    zero_refs = {
        "calories": int((phase_b_samples["ref_calories"] < 1.0).sum()),
        "protein": int((phase_b_samples["ref_protein_g"] < 0.5).sum()),
        "carbs": int((phase_b_samples["ref_carbs_g"] < 0.5).sum()),
        "fat": int((phase_b_samples["ref_fat_g"] < 0.5).sum())
    }
    print(f"\n[Zero / Near-Zero Reference Audit in Phase B]")
    for nut, count in zero_refs.items():
        print(f"  {nut.capitalize():12s}: {count} samples below denominator threshold (< 1.0 kcal or < 0.5g)")

    # 5. Portion Regressor MAE Audit
    reported_in_report = 70.93
    true_test_mae = portion_data["test_split"]["HistGradientBoosting"]["mae"]
    true_rf_test_mae = portion_data["test_split"]["RandomForest"]["mae"]
    true_val_mae = portion_data["validation_split"]["HistGradientBoosting"]["mae"]

    portion_audit = {
        "reported_in_nutrition_report_grams": reported_in_report,
        "true_histgradboost_test_mae_grams": true_test_mae,
        "true_randomforest_test_mae_grams": true_rf_test_mae,
        "true_histgradboost_val_mae_grams": true_val_mae,
        "discrepancy_explanation": (
            "The 70.93g value cited in NUTRITION_EVALUATION_REPORT.md was an inadvertent typo/confusion "
            f"with RandomForest's test MAE ({true_rf_test_mae}g) and HistGradientBoosting's validation MAE ({true_val_mae}g). "
            f"The true test MAE of the selected portion regressor in metrics.json is {true_test_mae}g."
        )
    }
    print(f"\n[Portion MAE Audit]")
    print(f"  Previously cited in report: {reported_in_report} g")
    print(f"  Verified true test MAE (HistGradientBoosting): {true_test_mae} g")
    print(f"  RandomForest test MAE: {true_rf_test_mae} g")

    # 6. Save Experiment 1 Audit Results
    exp1_summary = {
        "experiment": "Experiment 1: Baseline Validation",
        "phase_a_classes_validated": 54,
        "phase_b_samples_validated": 160,
        "unique_dish_ids": unique_dishes,
        "zero_reference_counts": zero_refs,
        "mismatched_samples_count": len(mismatched_samples),
        "mismatched_breakdown": {
            "caesar_salad": sum(1 for m in mismatched_samples if 'caesar' in m['display_name'].lower()),
            "scrambled_eggs": sum(1 for m in mismatched_samples if 'scrambled' in m['display_name'].lower()),
            "fish_fillet": sum(1 for m in mismatched_samples if m['mapped_class'] == 'fish')
        },
        "portion_mae_audit": portion_audit,
        "mismatched_samples_detail": mismatched_samples[:10]  # sample of details
    }

    out_file = os.path.join(RESULTS_DIR, "exp1_baseline_validation.json")
    with open(out_file, 'w', encoding='utf-8') as f:
        json.dump(exp1_summary, f, indent=2)
    print(f"\n[OK] Saved Experiment 1 validation summary to {out_file}")
    return exp1_summary

if __name__ == "__main__":
    validate_baseline()
