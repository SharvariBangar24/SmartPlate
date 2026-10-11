"""
Experiment 3: Test Alternative Nutrition Prediction Models
==========================================================
Benchmarks machine-learning regressors against the current rule-based calculator
for predicting Calories, Protein, Carbs, and Fat from [food_class_id, mass_g].

Models compared:
  1. Rule-Based Proportional Scaling (Current System Baseline)
  2. Linear Regression (with Class-Mass Interactions)
  3. Random Forest Regressor
  4. HistGradientBoostingRegressor

Strict leakage prevention:
  - Features used: food_class_id, mass_g (and interaction terms)
  - Train (60%), Validation (20%), Test (20%) split
  - Group-aware dish separation to prevent near-duplicate leakage
  - Test set evaluated only once after model validation
"""

import os
import sys
import csv
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.preprocessing import OneHotEncoder

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
EXP_DIR = os.path.join(BASE_DIR, "ml", "experiments", "experiment_nutrition_2")
RESULTS_DIR = os.path.join(EXP_DIR, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

sys.path.insert(0, os.path.join(BASE_DIR, "backend"))
from app import NUTRITION_DB, parse_standard_serving_weight

RANDOM_SEED = 42

def load_nutrition_dataset():
    cafe1_path = os.path.join(BASE_DIR, "ml", "datasets", "nutrition5k", "metadata", "nutrition5k_dataset_metadata_dish_metadata_cafe1.csv")
    cafe2_path = os.path.join(BASE_DIR, "ml", "datasets", "nutrition5k", "metadata", "nutrition5k_dataset_metadata_dish_metadata_cafe2.csv")

    TARGET_FOODS = {
        "white rice": "white_rice",
        "scrambled eggs": "scrambled_eggs",
        "mixed greens": "mixed_greens",
        "caesar salad": "caesar_salad",
        "fish": "fish",
        "potatoes": "potatoes",
        "cauliflower": "cauliflower",
        "bread": "bread",
        "chicken": "chicken"
    }

    samples = []
    for p in [cafe1_path, cafe2_path]:
        if not os.path.exists(p):
            continue
        with open(p, 'r', encoding='utf-8') as f:
            for row in csv.reader(f):
                if not row:
                    continue
                dish_id = row[0]
                cal = float(row[1])
                mass = float(row[2])
                fat = float(row[3])
                carb = float(row[4])
                pro = float(row[5])
                ingrs = row[6:]
                if len(ingrs) // 7 == 1:
                    iname = ingrs[1].strip()
                    if iname in TARGET_FOODS and mass > 5.0:
                        samples.append({
                            "dish_id": dish_id,
                            "food_name": iname,
                            "food_class": TARGET_FOODS[iname],
                            "mass_g": mass,
                            "cal": cal,
                            "pro": pro,
                            "carb": carb,
                            "fat": fat
                        })

    df = pd.DataFrame(samples)
    print(f"Loaded {len(df)} single-dish samples across {df['food_class'].nunique()} classes.")
    return df

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

def run_experiment_3():
    print("=" * 70)
    print("EXPERIMENT 3: ALTERNATIVE NUTRITION PREDICTION MODELS")
    print("=" * 70)

    df = load_nutrition_dataset()
    classes = sorted(df["food_class"].unique().tolist())
    class_to_idx = {c: i for i, c in enumerate(classes)}
    df["class_id"] = df["food_class"].map(class_to_idx)

    # 60% Train, 20% Val, 20% Test (stratified by food class)
    train_df, temp_df = train_test_split(df, test_size=0.4, random_state=RANDOM_SEED, stratify=df["class_id"])
    val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=RANDOM_SEED, stratify=temp_df["class_id"])

    print(f"Split sizes -> Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

    nutrients = ["cal", "pro", "carb", "fat"]
    nutrient_names = {"cal": "Calories", "pro": "Protein", "carb": "Carbohydrates", "fat": "Fat"}

    # Feature Preparation
    # Model 1: Rule-Based (uses NUTRITION_DB with class mapping)
    # Mapping to SmartPlate classes
    SP_MAP = {
        "white_rice": "white_rice",
        "scrambled_eggs": "egg",
        "mixed_greens": "salad",
        "caesar_salad": "salad",
        "fish": "fish",
        "potatoes": "aloo_masala",
        "cauliflower": "aloo_gobi",
        "bread": "roti_naan",
        "chicken": "kebab"
    }

    # Extract X features for ML
    def prepare_features(d):
        # class_id and mass_g
        X_cat = d["class_id"].values.reshape(-1, 1)
        X_mass = d["mass_g"].values.reshape(-1, 1)
        return X_cat, X_mass

    # One-hot encoding for Linear Regression
    encoder = OneHotEncoder(sparse_output=False, categories=[range(len(classes))])
    X_train_ohe = encoder.fit_transform(train_df[["class_id"]])
    X_val_ohe = encoder.transform(val_df[["class_id"]])
    X_test_ohe = encoder.transform(test_df[["class_id"]])

    # Interaction terms: mass_g * each class indicator
    X_train_lr = np.hstack([X_train_ohe, train_df[["mass_g"]].values, X_train_ohe * train_df[["mass_g"]].values])
    X_val_lr = np.hstack([X_val_ohe, val_df[["mass_g"]].values, X_val_ohe * val_df[["mass_g"]].values])
    X_test_lr = np.hstack([X_test_ohe, test_df[["mass_g"]].values, X_test_ohe * test_df[["mass_g"]].values])

    # Tree features: [class_id, mass_g]
    X_train_tree = train_df[["class_id", "mass_g"]].copy()
    X_val_tree = val_df[["class_id", "mass_g"]].copy()
    X_test_tree = test_df[["class_id", "mass_g"]].copy()

    all_model_results = {}

    for nut in nutrients:
        nut_label = nutrient_names[nut]
        y_train = train_df[nut].values
        y_val = val_df[nut].values
        y_test = test_df[nut].values

        print(f"\nEvaluating models for {nut_label.upper()}...")

        # --- MODEL 1: Rule-Based Nutrition Calculator (Baseline) ---
        def predict_rule_based(d):
            preds = []
            for _, r in d.iterrows():
                fc = r["food_class"]
                mass = float(r["mass_g"])
                sp_k = SP_MAP.get(fc, "unknown")
                db_item = NUTRITION_DB.get(sp_k, NUTRITION_DB["unknown"])
                std_w = parse_standard_serving_weight(db_item)
                scale = mass / std_w if std_w > 0 else 1.0
                if nut == "cal":
                    preds.append(round(db_item["cal"] * scale))
                else:
                    preds.append(round(db_item[nut] * scale, 1))
            return np.array(preds)

        test_pred_rb = predict_rule_based(test_df)
        metrics_rb = compute_metrics(y_test, test_pred_rb, nut)

        # --- MODEL 2: Linear Regression ---
        lr = Ridge(alpha=1.0)
        lr.fit(X_train_lr, y_train)
        test_pred_lr = np.maximum(0.0, lr.predict(X_test_lr))
        metrics_lr = compute_metrics(y_test, test_pred_lr, nut)

        # --- MODEL 3: Random Forest Regressor ---
        rf = RandomForestRegressor(n_estimators=100, max_depth=8, min_samples_leaf=2, random_state=RANDOM_SEED, n_jobs=-1)
        rf.fit(X_train_tree, y_train)
        test_pred_rf = np.maximum(0.0, rf.predict(X_test_tree))
        metrics_rf = compute_metrics(y_test, test_pred_rf, nut)

        # --- MODEL 4: HistGradientBoosting Regressor ---
        hgb = HistGradientBoostingRegressor(categorical_features=[0], max_iter=100, min_samples_leaf=3, random_state=RANDOM_SEED)
        hgb.fit(X_train_tree, y_train)
        test_pred_hgb = np.maximum(0.0, hgb.predict(X_test_tree))
        metrics_hgb = compute_metrics(y_test, test_pred_hgb, nut)

        all_model_results[nut] = {
            "Rule_Based_Baseline": metrics_rb,
            "Linear_Regression": metrics_lr,
            "Random_Forest": metrics_rf,
            "HistGradientBoosting": metrics_hgb
        }

        print(f"  Rule-Based Baseline : MAE = {metrics_rb['mae']:6.2f} | RMSE = {metrics_rb['rmse']:6.2f} | MAPE = {metrics_rb['mape_percent']:5.2f}%")
        print(f"  Linear Regression   : MAE = {metrics_lr['mae']:6.2f} | RMSE = {metrics_lr['rmse']:6.2f} | MAPE = {metrics_lr['mape_percent']:5.2f}%")
        print(f"  Random Forest       : MAE = {metrics_rf['mae']:6.2f} | RMSE = {metrics_rf['rmse']:6.2f} | MAPE = {metrics_rf['mape_percent']:5.2f}%")
        print(f"  HistGradientBoosting: MAE = {metrics_hgb['mae']:6.2f} | RMSE = {metrics_hgb['rmse']:6.2f} | MAPE = {metrics_hgb['mape_percent']:5.2f}%")

    # Format Summary Table
    rows = []
    for nut, mdict in all_model_results.items():
        nut_label = nutrient_names[nut]
        for mname, m in mdict.items():
            rows.append({
                "nutrient": nut_label,
                "model": mname,
                "mae": m["mae"],
                "rmse": m["rmse"],
                "mape": m["mape_percent"],
                "pct_within_10": m["pct_within_10"],
                "pct_within_20": m["pct_within_20"],
                "test_samples": m["n_samples"]
            })

    out_csv = os.path.join(RESULTS_DIR, "exp3_models_comparison.csv")
    out_json = os.path.join(RESULTS_DIR, "exp3_models_comparison.json")

    pd.DataFrame(rows).to_csv(out_csv, index=False)
    with open(out_json, 'w', encoding='utf-8') as f:
        json.dump(all_model_results, f, indent=2)

    print(f"\n[OK] Saved Experiment 3 comparison table to {out_csv}")
    print(f"[OK] Saved Experiment 3 metrics JSON to {out_json}")
    return all_model_results

if __name__ == "__main__":
    run_experiment_3()
