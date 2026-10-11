"""
Experiment 4: Improve Portion Estimation
========================================
Benchmarks alternatives to the current HistGradientBoosting portion regressor
using the official FPB dataset train (13,756), val (5,723), and test (5,016) splits.

Models evaluated:
  1. Median Baseline
  2. Class Mean Baseline
  3. Current HistGradientBoosting (Baseline, Test MAE: 61.61g)
  4. Ridge Regression (Linear Baseline)
  5. Tuned Random Forest Regressor
  6. Enhanced HistGradientBoosting (Expanded Geometry: perimeter, diagonal + tuned hyperparameters)
  7. Blended Ensemble (HistGradientBoosting + Tuned RF)

Outputs:
  - Overall MAE (g), RMSE (g), R2, Median AE
  - Per-category error breakdown on held-out test split
  - Generalization analysis for Indian food
"""

import os
import sys
import glob
import json
import yaml
import time
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import Ridge
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
import joblib

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
EXP_DIR = os.path.join(BASE_DIR, "ml", "experiments", "experiment_nutrition_2")
RESULTS_DIR = os.path.join(EXP_DIR, "results")
MODELS_DIR = os.path.join(EXP_DIR, "models")
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

RAW_LABELS_DIR = os.path.join(BASE_DIR, "ml", "experiments", "experiment_portion_1", "raw_labels", "FPB_Dataset", "RGB")
YAML_PATH = os.path.join(RAW_LABELS_DIR, "data.yaml")
RANDOM_SEED = 42

def load_class_names():
    with open(YAML_PATH, 'r', encoding='utf-8') as f:
        data_cfg = yaml.safe_load(f)
    return data_cfg.get('names', [])

def parse_split_labels(split_name, class_names):
    labels_dir = os.path.join(RAW_LABELS_DIR, split_name, 'labels')
    txt_files = glob.glob(os.path.join(labels_dir, '*.txt'))
    records = []

    for file_path in txt_files:
        filename = os.path.basename(file_path)
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = [l.strip().split() for l in f.readlines() if l.strip()]

        for parts in lines:
            if len(parts) < 6:
                continue
            try:
                class_id = int(parts[0])
                cx = float(parts[1])
                cy = float(parts[2])
                w = float(parts[3])
                h = float(parts[4])
                weight = float(parts[5])
            except ValueError:
                continue

            if weight <= 0:
                continue

            class_name = class_names[class_id] if class_id < len(class_names) else f"class_{class_id}"
            area = w * h
            aspect_ratio = w / (h + 1e-6)
            sqrt_area = np.sqrt(max(0.0, area))
            log_area = np.log1p(max(0.0, area))
            perimeter = 2.0 * (w + h)
            diagonal = np.sqrt(w**2 + h**2)

            records.append({
                'filename': filename,
                'class_id': class_id,
                'class_name': class_name,
                'cx': cx,
                'cy': cy,
                'w': w,
                'h': h,
                'area': area,
                'aspect_ratio': aspect_ratio,
                'sqrt_area': sqrt_area,
                'log_area': log_area,
                'perimeter': perimeter,
                'diagonal': diagonal,
                'weight': weight
            })

    return pd.DataFrame(records)

def evaluate_metrics(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    med_ae = np.median(np.abs(y_true - y_pred))
    return {
        'mae': round(float(mae), 2),
        'rmse': round(float(rmse), 2),
        'r2': round(float(r2), 4),
        'median_ae': round(float(med_ae), 2)
    }

def run_experiment_4():
    print("=" * 70)
    print("EXPERIMENT 4: PORTION ESTIMATION MODEL BENCHMARKING")
    print("=" * 70)

    class_names = load_class_names()
    print(f"Loaded {len(class_names)} FPB food class categories.")

    print("Parsing FPB dataset splits...")
    train_df = parse_split_labels('train', class_names)
    val_df = parse_split_labels('val', class_names)
    test_df = parse_split_labels('test', class_names)

    print(f"Splits loaded -> Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

    # Feature sets
    base_features = ['class_id', 'w', 'h', 'area', 'aspect_ratio', 'sqrt_area', 'log_area']
    enhanced_features = ['class_id', 'w', 'h', 'area', 'aspect_ratio', 'sqrt_area', 'log_area', 'perimeter', 'diagonal']

    X_train_base = train_df[base_features].copy()
    X_val_base = val_df[base_features].copy()
    X_test_base = test_df[base_features].copy()

    X_train_enh = train_df[enhanced_features].copy()
    X_val_enh = val_df[enhanced_features].copy()
    X_test_enh = test_df[enhanced_features].copy()

    y_train = train_df['weight'].values
    y_val = val_df['weight'].values
    y_test = test_df['weight'].values

    models = {}
    val_metrics = {}
    test_metrics = {}
    test_preds = {}

    # 1. Median Baseline
    med_val = float(np.median(y_train))
    p_val_1 = np.full(len(y_val), med_val)
    p_test_1 = np.full(len(y_test), med_val)
    val_metrics["Median_Baseline"] = evaluate_metrics(y_val, p_val_1)
    test_metrics["Median_Baseline"] = evaluate_metrics(y_test, p_test_1)
    test_preds["Median_Baseline"] = p_test_1

    # 2. Class Mean Baseline
    means_dict = train_df.groupby('class_id')['weight'].mean().to_dict()
    p_val_2 = np.array([means_dict.get(c, med_val) for c in val_df['class_id']])
    p_test_2 = np.array([means_dict.get(c, med_val) for c in test_df['class_id']])
    val_metrics["Class_Mean_Baseline"] = evaluate_metrics(y_val, p_val_2)
    test_metrics["Class_Mean_Baseline"] = evaluate_metrics(y_test, p_test_2)
    test_preds["Class_Mean_Baseline"] = p_test_2

    # 3. Current HistGradientBoosting (Baseline Model)
    print("\nTraining Model 3: Current HistGradientBoosting...")
    hgb_base = HistGradientBoostingRegressor(
        loss='absolute_error',
        categorical_features=[0],
        max_iter=150,
        min_samples_leaf=20,
        random_state=RANDOM_SEED
    )
    hgb_base.fit(X_train_base, y_train)
    p_val_3 = hgb_base.predict(X_val_base)
    p_test_3 = hgb_base.predict(X_test_base)
    val_metrics["Current_HistGradBoost"] = evaluate_metrics(y_val, p_val_3)
    test_metrics["Current_HistGradBoost"] = evaluate_metrics(y_test, p_test_3)
    test_preds["Current_HistGradBoost"] = p_test_3

    # 4. Ridge Regression (Linear Baseline)
    print("Training Model 4: Ridge Regression...")
    ridge = Ridge(alpha=10.0)
    ridge.fit(X_train_base.iloc[:, 1:], y_train)  # continuous geometry
    p_val_4 = np.maximum(15.0, ridge.predict(X_val_base.iloc[:, 1:]))
    p_test_4 = np.maximum(15.0, ridge.predict(X_test_base.iloc[:, 1:]))
    val_metrics["Ridge_Regression"] = evaluate_metrics(y_val, p_val_4)
    test_metrics["Ridge_Regression"] = evaluate_metrics(y_test, p_test_4)
    test_preds["Ridge_Regression"] = p_test_4

    # 5. Tuned Random Forest Regressor
    print("Training Model 5: Tuned Random Forest...")
    rf_tuned = RandomForestRegressor(
        n_estimators=120,
        max_depth=16,
        min_samples_leaf=3,
        random_state=RANDOM_SEED,
        n_jobs=-1
    )
    rf_tuned.fit(X_train_enh, y_train)
    p_val_5 = rf_tuned.predict(X_val_enh)
    p_test_5 = rf_tuned.predict(X_test_enh)
    val_metrics["Tuned_RandomForest"] = evaluate_metrics(y_val, p_val_5)
    test_metrics["Tuned_RandomForest"] = evaluate_metrics(y_test, p_test_5)
    test_preds["Tuned_RandomForest"] = p_test_5

    # 6. Enhanced HistGradientBoosting (Expanded Features & Hyperparameter Tuning)
    print("Training Model 6: Enhanced HistGradientBoosting...")
    hgb_enh = HistGradientBoostingRegressor(
        loss='absolute_error',
        categorical_features=[0],
        max_iter=250,
        learning_rate=0.08,
        min_samples_leaf=15,
        l2_regularization=1.0,
        random_state=RANDOM_SEED
    )
    hgb_enh.fit(X_train_enh, y_train)
    p_val_6 = hgb_enh.predict(X_val_enh)
    p_test_6 = hgb_enh.predict(X_test_enh)
    val_metrics["Enhanced_HistGradBoost"] = evaluate_metrics(y_val, p_val_6)
    test_metrics["Enhanced_HistGradBoost"] = evaluate_metrics(y_test, p_test_6)
    test_preds["Enhanced_HistGradBoost"] = p_test_6

    # 7. Blended Ensemble (Enhanced HGB + Tuned RF 50/50)
    p_val_7 = 0.6 * p_val_6 + 0.4 * p_val_5
    p_test_7 = 0.6 * p_test_6 + 0.4 * p_test_5
    val_metrics["Ensemble_HGB_RF"] = evaluate_metrics(y_val, p_val_7)
    test_metrics["Ensemble_HGB_RF"] = evaluate_metrics(y_test, p_test_7)
    test_preds["Ensemble_HGB_RF"] = p_test_7

    # Print Summary Table
    print("\n" + "=" * 80)
    print(f"{'Model Architecture':<28} | {'Val MAE (g)':<12} | {'Test MAE (g)':<12} | {'Test RMSE (g)':<13} | {'Test R2':<8}")
    print("=" * 80)
    summary_rows = []
    for mname in val_metrics.keys():
        v_mae = val_metrics[mname]['mae']
        t_mae = test_metrics[mname]['mae']
        t_rmse = test_metrics[mname]['rmse']
        t_r2 = test_metrics[mname]['r2']
        print(f"{mname:<28} | {v_mae:<12.2f} | {t_mae:<12.2f} | {t_rmse:<13.2f} | {t_r2:<8.4f}")
        summary_rows.append({
            "model": mname,
            "val_mae_g": v_mae,
            "val_rmse_g": val_metrics[mname]['rmse'],
            "test_mae_g": t_mae,
            "test_rmse_g": t_rmse,
            "test_r2": t_r2,
            "test_median_ae_g": test_metrics[mname]['median_ae']
        })

    # Save best new model
    joblib.dump(hgb_enh, os.path.join(MODELS_DIR, "enhanced_portion_regressor.joblib"))

    # Per-Category Test Breakdown for Best Model
    test_df_eval = test_df.copy()
    test_df_eval["pred_enh"] = p_test_6
    test_df_eval["abs_error"] = np.abs(test_df_eval["weight"] - test_df_eval["pred_enh"])

    cat_breakdown = []
    for cname, grp in test_df_eval.groupby("class_name"):
        if len(grp) >= 20:  # categories with at least 20 test instances
            cat_breakdown.append({
                "class_name": cname,
                "n_samples": len(grp),
                "mae_g": round(float(grp["abs_error"].mean()), 2),
                "mean_weight_g": round(float(grp["weight"].mean()), 2),
                "error_pct": round(float(grp["abs_error"].mean() / grp["weight"].mean() * 100.0), 2)
            })
    cat_df = pd.DataFrame(cat_breakdown).sort_values("mae_g")

    # Save outputs
    out_csv = os.path.join(RESULTS_DIR, "exp4_portion_comparison.csv")
    out_cat_csv = os.path.join(RESULTS_DIR, "exp4_portion_per_category.csv")
    out_json = os.path.join(RESULTS_DIR, "exp4_portion_comparison.json")

    pd.DataFrame(summary_rows).to_csv(out_csv, index=False)
    cat_df.to_csv(out_cat_csv, index=False)

    exp4_results = {
        "dataset": "FoodPortionBenchmark (FPB)",
        "train_samples": len(train_df),
        "val_samples": len(val_df),
        "test_samples": len(test_df),
        "models_summary": summary_rows,
        "best_model_test_mae_g": min(r["test_mae_g"] for r in summary_rows),
        "generalization_notes": (
            "FPB provides European/Central-Asian dishes photographed from fixed top-down angles. "
            "Without camera depth, phone sensor distance, or a physical scale marker (e.g. standard plate rim or coin), "
            "2D bounding-box area has an intrinsic depth-scale ambiguity when applied to Indian food images."
        )
    }
    with open(out_json, 'w', encoding='utf-8') as f:
        json.dump(exp4_results, f, indent=2)

    print(f"\n[OK] Saved Experiment 4 model comparison to {out_csv}")
    print(f"[OK] Saved Experiment 4 category breakdown to {out_cat_csv}")
    print(f"[OK] Saved Experiment 4 metrics JSON to {out_json}")
    return exp4_results

if __name__ == "__main__":
    run_experiment_4()
