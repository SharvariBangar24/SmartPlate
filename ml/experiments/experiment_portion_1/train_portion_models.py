import os
import sys
import glob
import json
import yaml
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
import joblib

# Set random seed for reproducibility
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RAW_LABELS_DIR = os.path.join(BASE_DIR, 'raw_labels', 'FPB_Dataset', 'RGB')
OUTPUT_MODELS_DIR = os.path.join(BASE_DIR, 'models')
os.makedirs(OUTPUT_MODELS_DIR, exist_ok=True)

YAML_PATH = os.path.join(RAW_LABELS_DIR, 'data.yaml')

def load_class_names():
    if not os.path.exists(YAML_PATH):
        raise FileNotFoundError(f"data.yaml not found at {YAML_PATH}")
    with open(YAML_PATH, 'r', encoding='utf-8') as f:
        data_cfg = yaml.safe_load(f)
    return data_cfg.get('names', [])

def parse_split_labels(split_name, class_names):
    labels_dir = os.path.join(RAW_LABELS_DIR, split_name, 'labels')
    if not os.path.exists(labels_dir):
        raise FileNotFoundError(f"Labels directory not found: {labels_dir}")

    txt_files = glob.glob(os.path.join(labels_dir, '*.txt'))
    print(f"[{split_name.upper()}] Found {len(txt_files)} label files in {labels_dir}")

    records = []
    total_raw_boxes = 0
    discarded_boxes = 0

    for file_path in txt_files:
        filename = os.path.basename(file_path)
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = [l.strip().split() for l in f.readlines() if l.strip()]

        for parts in lines:
            total_raw_boxes += 1
            if len(parts) < 6:
                discarded_boxes += 1
                continue

            try:
                class_id = int(parts[0])
                cx = float(parts[1])
                cy = float(parts[2])
                w = float(parts[3])
                h = float(parts[4])
                weight = float(parts[5])
            except ValueError:
                discarded_boxes += 1
                continue

            # Strict filter: Discard unmeasured / missing / negative weights
            if weight <= 0:
                discarded_boxes += 1
                continue

            class_name = class_names[class_id] if class_id < len(class_names) else f"class_{class_id}"
            area = w * h
            aspect_ratio = w / (h + 1e-6)
            sqrt_area = np.sqrt(max(0.0, area))
            log_area = np.log1p(max(0.0, area))

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
                'weight': weight
            })

    df = pd.DataFrame(records)
    print(f"[{split_name.upper()}] Raw BBoxes: {total_raw_boxes} | Discarded (<= 0): {discarded_boxes} | Usable: {len(df)}")
    return df, total_raw_boxes, discarded_boxes

# Approach A: Median Baseline
class MedianBaseline:
    def __init__(self):
        self.median_val = 0.0

    def fit(self, y):
        self.median_val = float(np.median(y))

    def predict(self, X):
        return np.full(len(X), self.median_val)

# Approach B: Class-Conditional Mean Baseline with Fallback
class ClassMeanBaseline:
    def __init__(self):
        self.class_means = {}
        self.global_median = 0.0

    def fit(self, class_ids, y):
        self.global_median = float(np.median(y))
        df_tmp = pd.DataFrame({'class_id': class_ids, 'y': y})
        self.class_means = df_tmp.groupby('class_id')['y'].mean().to_dict()

    def predict(self, class_ids):
        return np.array([self.class_means.get(c, self.global_median) for c in class_ids])

def evaluate_predictions(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    med_ae = np.median(np.abs(y_true - y_pred))
    return {
        'mae': round(float(mae), 3),
        'rmse': round(float(rmse), 3),
        'r2': round(float(r2), 4),
        'median_ae': round(float(med_ae), 3)
    }

def main():
    start_time = time.time()
    print("=" * 70)
    print("Smart Plate - Portion Estimation Experiment 1 (FPB)")
    print("=" * 70)

    class_names = load_class_names()
    print(f"Loaded {len(class_names)} food class categories from data.yaml")

    # 1. Parse and filter splits
    train_df, train_raw, train_disc = parse_split_labels('train', class_names)
    val_df, val_raw, val_disc = parse_split_labels('val', class_names)
    test_df, test_raw, test_disc = parse_split_labels('test', class_names)

    # Dataset Summary Statistics
    dataset_summary = {
        'license': 'CC BY-NC 4.0',
        'source': 'Institute of Smart Systems and Artificial Intelligence (ISSAI) - FPB',
        'num_classes': len(class_names),
        'splits': {
            'train': {
                'raw_bboxes': train_raw,
                'discarded_negative_or_zero': train_disc,
                'usable_samples': len(train_df),
                'unique_images': train_df['filename'].nunique(),
                'weight_stats': {
                    'min': round(float(train_df['weight'].min()), 2),
                    'median': round(float(train_df['weight'].median()), 2),
                    'mean': round(float(train_df['weight'].mean()), 2),
                    'max': round(float(train_df['weight'].max()), 2),
                    'std': round(float(train_df['weight'].std()), 2)
                }
            },
            'val': {
                'raw_bboxes': val_raw,
                'discarded_negative_or_zero': val_disc,
                'usable_samples': len(val_df),
                'unique_images': val_df['filename'].nunique(),
                'weight_stats': {
                    'min': round(float(val_df['weight'].min()), 2),
                    'median': round(float(val_df['weight'].median()), 2),
                    'mean': round(float(val_df['weight'].mean()), 2),
                    'max': round(float(val_df['weight'].max()), 2),
                    'std': round(float(val_df['weight'].std()), 2)
                }
            },
            'test': {
                'raw_bboxes': test_raw,
                'discarded_negative_or_zero': test_disc,
                'usable_samples': len(test_df),
                'unique_images': test_df['filename'].nunique(),
                'weight_stats': {
                    'min': round(float(test_df['weight'].min()), 2),
                    'median': round(float(test_df['weight'].median()), 2),
                    'mean': round(float(test_df['weight'].mean()), 2),
                    'max': round(float(test_df['weight'].max()), 2),
                    'std': round(float(test_df['weight'].std()), 2)
                }
            }
        }
    }

    with open(os.path.join(BASE_DIR, 'dataset_summary.json'), 'w', encoding='utf-8') as f:
        json.dump(dataset_summary, f, indent=2)

    # 2. Extract Features (Inference-Available Only)
    feature_cols = ['class_id', 'w', 'h', 'area', 'aspect_ratio', 'sqrt_area', 'log_area']
    
    X_train = train_df[feature_cols].copy()
    y_train = train_df['weight'].values

    X_val = val_df[feature_cols].copy()
    y_val = val_df['weight'].values

    X_test = test_df[feature_cols].copy()
    y_test = test_df['weight'].values

    print("\nTraining Portion Models on features:", feature_cols)

    # 3. Model A: Training-set Median Baseline
    model_a = MedianBaseline()
    model_a.fit(y_train)
    val_pred_a = model_a.predict(X_val)
    metrics_a_val = evaluate_predictions(y_val, val_pred_a)
    print(f"Model A (Train Median) -> Val MAE: {metrics_a_val['mae']} g | RMSE: {metrics_a_val['rmse']} g")

    # 4. Model B: Class-Conditional Mean Baseline
    model_b = ClassMeanBaseline()
    model_b.fit(train_df['class_id'].values, y_train)
    val_pred_b = model_b.predict(val_df['class_id'].values)
    metrics_b_val = evaluate_predictions(y_val, val_pred_b)
    print(f"Model B (Class Mean)   -> Val MAE: {metrics_b_val['mae']} g | RMSE: {metrics_b_val['rmse']} g")

    # 5. Model C1: HistGradientBoostingRegressor (with native categorical support for class_id)
    # class_id is column 0
    model_c1 = HistGradientBoostingRegressor(
        loss='absolute_error',
        categorical_features=[0],
        max_iter=150,
        min_samples_leaf=20,
        random_state=RANDOM_SEED
    )
    model_c1.fit(X_train, y_train)
    val_pred_c1 = model_c1.predict(X_val)
    metrics_c1_val = evaluate_predictions(y_val, val_pred_c1)
    print(f"Model C1 (HistGradBoost) -> Val MAE: {metrics_c1_val['mae']} g | RMSE: {metrics_c1_val['rmse']} g | R2: {metrics_c1_val['r2']}")

    # 6. Model C2: Random Forest Regressor
    model_c2 = RandomForestRegressor(
        n_estimators=100,
        max_depth=14,
        min_samples_leaf=5,
        random_state=RANDOM_SEED,
        n_jobs=-1
    )
    model_c2.fit(X_train, y_train)
    val_pred_c2 = model_c2.predict(X_val)
    metrics_c2_val = evaluate_predictions(y_val, val_pred_c2)
    print(f"Model C2 (RandomForest) -> Val MAE: {metrics_c2_val['mae']} g | RMSE: {metrics_c2_val['rmse']} g | R2: {metrics_c2_val['r2']}")

    # Compare validation performances and pick best
    val_candidates = {
        'Median_Baseline': (model_a, metrics_a_val, val_pred_a),
        'Class_Mean_Baseline': (model_b, metrics_b_val, val_pred_b),
        'HistGradientBoosting': (model_c1, metrics_c1_val, val_pred_c1),
        'RandomForest': (model_c2, metrics_c2_val, val_pred_c2)
    }

    best_model_name = min(val_candidates.keys(), key=lambda k: val_candidates[k][1]['mae'])
    best_model, best_val_metrics, best_val_pred = val_candidates[best_model_name]

    print(f"\n>>> Best Validation Model: {best_model_name} (Val MAE: {best_val_metrics['mae']} g) <<<")

    # 7. Evaluate on held-out test split
    if best_model_name == 'Median_Baseline':
        test_pred_best = best_model.predict(X_test)
    elif best_model_name == 'Class_Mean_Baseline':
        test_pred_best = best_model.predict(test_df['class_id'].values)
    else:
        test_pred_best = best_model.predict(X_test)

    test_metrics_best = evaluate_predictions(y_test, test_pred_best)
    print(f"FINAL TEST EVALUATION ({best_model_name}): MAE = {test_metrics_best['mae']} g | RMSE = {test_metrics_best['rmse']} g | R2 = {test_metrics_best['r2']}")

    # Also compute test metrics for all models for comprehensive comparison table
    test_pred_a = model_a.predict(X_test)
    test_pred_b = model_b.predict(test_df['class_id'].values)
    test_pred_c1 = model_c1.predict(X_test)
    test_pred_c2 = model_c2.predict(X_test)

    metrics_all = {
        'validation_split': {
            'Median_Baseline': metrics_a_val,
            'Class_Mean_Baseline': metrics_b_val,
            'HistGradientBoosting': metrics_c1_val,
            'RandomForest': metrics_c2_val
        },
        'test_split': {
            'Median_Baseline': evaluate_predictions(y_test, test_pred_a),
            'Class_Mean_Baseline': evaluate_predictions(y_test, test_pred_b),
            'HistGradientBoosting': evaluate_predictions(y_test, test_pred_c1),
            'RandomForest': evaluate_predictions(y_test, test_pred_c2)
        },
        'best_model': best_model_name,
        'final_test_mae_grams': test_metrics_best['mae'],
        'final_test_rmse_grams': test_metrics_best['rmse'],
        'final_test_r2': test_metrics_best['r2'],
        'sample_counts': {
            'train_usable': len(train_df),
            'val_usable': len(val_df),
            'test_usable': len(test_df)
        }
    }

    with open(os.path.join(BASE_DIR, 'metrics.json'), 'w', encoding='utf-8') as f:
        json.dump(metrics_all, f, indent=2)

    # 8. Save Test Predictions CSV
    test_out_df = test_df[['filename', 'class_id', 'class_name', 'w', 'h', 'area', 'aspect_ratio', 'weight']].copy()
    test_out_df.rename(columns={'weight': 'actual_weight_grams'}, inplace=True)
    test_out_df['pred_median_grams'] = np.round(test_pred_a, 2)
    test_out_df['pred_class_mean_grams'] = np.round(test_pred_b, 2)
    test_out_df['pred_hist_gb_grams'] = np.round(test_pred_c1, 2)
    test_out_df['pred_rf_grams'] = np.round(test_pred_c2, 2)
    test_out_df['best_model_pred_grams'] = np.round(test_pred_best, 2)
    test_out_df['residual_error_grams'] = np.round(test_out_df['best_model_pred_grams'] - test_out_df['actual_weight_grams'], 2)
    test_out_df['abs_error_grams'] = np.round(np.abs(test_out_df['residual_error_grams']), 2)

    pred_csv_path = os.path.join(BASE_DIR, 'test_predictions.csv')
    test_out_df.to_csv(pred_csv_path, index=False)
    print(f"Saved test predictions to {pred_csv_path}")

    # 9. Save Best Model
    saved_model_path = os.path.join(OUTPUT_MODELS_DIR, 'best_portion_regressor.joblib')
    joblib.dump(best_model, saved_model_path)
    print(f"Saved trained best model to {saved_model_path}")

    # 10. Generate Visual Comparison Chart
    plt.figure(figsize=(14, 5))

    # Subplot 1: Validation & Test MAE Comparison
    plt.subplot(1, 3, 1)
    models_list = ['Median', 'Class Mean', 'HistGradBoost', 'RandomForest']
    val_maes = [metrics_a_val['mae'], metrics_b_val['mae'], metrics_c1_val['mae'], metrics_c2_val['mae']]
    test_maes = [
        metrics_all['test_split']['Median_Baseline']['mae'],
        metrics_all['test_split']['Class_Mean_Baseline']['mae'],
        metrics_all['test_split']['HistGradientBoosting']['mae'],
        metrics_all['test_split']['RandomForest']['mae']
    ]

    x = np.arange(len(models_list))
    width = 0.35
    plt.bar(x - width/2, val_maes, width, label='Val MAE (g)', color='#4C72B0')
    plt.bar(x + width/2, test_maes, width, label='Test MAE (g)', color='#55A868')
    plt.ylabel('Mean Absolute Error (grams)')
    plt.title('MAE Comparison Across Models')
    plt.xticks(x, models_list, rotation=15)
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.5)

    # Subplot 2: Actual vs Predicted Scatter on Test
    plt.subplot(1, 3, 2)
    plt.scatter(y_test, test_pred_best, alpha=0.3, s=15, color='#2B5C8F')
    max_val = max(y_test.max(), test_pred_best.max()) + 20
    plt.plot([0, max_val], [0, max_val], 'r--', lw=1.5, label='Ideal y=x')
    plt.xlabel('Actual Weight (grams)')
    plt.ylabel('Predicted Weight (grams)')
    plt.title(f'Test Set: Actual vs Predicted ({best_model_name})')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)

    # Subplot 3: Residual Error Distribution
    plt.subplot(1, 3, 3)
    residuals = test_out_df['residual_error_grams'].values
    plt.hist(residuals, bins=40, color='#C44E52', edgecolor='black', alpha=0.7)
    plt.axvline(0, color='black', linestyle='--', lw=1.5)
    plt.xlabel('Residual Error (Predicted - Actual, grams)')
    plt.ylabel('Frequency')
    plt.title('Test Error Distribution')
    plt.grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()
    chart_path = os.path.join(BASE_DIR, 'comparison_chart.png')
    plt.savefig(chart_path, dpi=150)
    plt.close()
    print(f"Saved visual evaluation chart to {chart_path}")

    # 11. Configuration JSON
    config_dict = {
        'experiment_name': 'experiment_portion_1',
        'random_seed': RANDOM_SEED,
        'feature_columns': feature_cols,
        'models_trained': ['MedianBaseline', 'ClassMeanBaseline', 'HistGradientBoostingRegressor', 'RandomForestRegressor'],
        'dataset_source': 'issai/Food_Portion_Benchmark',
        'license': 'CC BY-NC 4.0',
        'training_time_seconds': round(time.time() - start_time, 2)
    }
    with open(os.path.join(BASE_DIR, 'config.json'), 'w', encoding='utf-8') as f:
        json.dump(config_dict, f, indent=2)

    # 12. Markdown Experiment Report
    report_content = f"""# Smart Plate - Portion Estimation Experiment 1 Report

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
- **Total Classes:** {len(class_names)} Central Asian / Continental food categories.
- **Filtering Rule:** Records with `weight <= 0` (unmeasured secondary items) were strictly discarded.

| Split | Raw Bounding Boxes | Discarded (`weight <= 0`) | Usable Samples | Unique Images | Mean Weight | Median Weight | Std Dev |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Train** | {train_raw:,} | {train_disc:,} | **{len(train_df):,}** | {train_df['filename'].nunique():,} | {train_df['weight'].mean():.1f} g | {train_df['weight'].median():.1f} g | {train_df['weight'].std():.1f} g |
| **Validation** | {val_raw:,} | {val_disc:,} | **{len(val_df):,}** | {val_df['filename'].nunique():,} | {val_df['weight'].mean():.1f} g | {val_df['weight'].median():.1f} g | {val_df['weight'].std():.1f} g |
| **Test** | {test_raw:,} | {test_disc:,} | **{len(test_df):,}** | {test_df['filename'].nunique():,} | {test_df['weight'].mean():.1f} g | {test_df['weight'].median():.1f} g | {test_df['weight'].std():.1f} g |

## 3. Model Comparison & Metrics

| Model Architecture | Validation MAE (g) | Validation RMSE (g) | Validation $R^2$ | Test MAE (g) | Test RMSE (g) | Test $R^2$ |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **A. Training Median Baseline** | {metrics_a_val['mae']} | {metrics_a_val['rmse']} | {metrics_a_val['r2']} | {metrics_all['test_split']['Median_Baseline']['mae']} | {metrics_all['test_split']['Median_Baseline']['rmse']} | {metrics_all['test_split']['Median_Baseline']['r2']} |
| **B. Class-Conditional Mean** | {metrics_b_val['mae']} | {metrics_b_val['rmse']} | {metrics_b_val['r2']} | {metrics_all['test_split']['Class_Mean_Baseline']['mae']} | {metrics_all['test_split']['Class_Mean_Baseline']['rmse']} | {metrics_all['test_split']['Class_Mean_Baseline']['r2']} |
| **C1. HistGradientBoosting** | **{metrics_c1_val['mae']}** | **{metrics_c1_val['rmse']}** | **{metrics_c1_val['r2']}** | **{metrics_all['test_split']['HistGradientBoosting']['mae']}** | **{metrics_all['test_split']['HistGradientBoosting']['rmse']}** | **{metrics_all['test_split']['HistGradientBoosting']['r2']}** |
| **C2. Random Forest** | {metrics_c2_val['mae']} | {metrics_c2_val['rmse']} | {metrics_c2_val['r2']} | {metrics_all['test_split']['RandomForest']['mae']} | {metrics_all['test_split']['RandomForest']['rmse']} | {metrics_all['test_split']['RandomForest']['r2']} |

- **Best Validation Model:** `{best_model_name}` (Val MAE: **{best_val_metrics['mae']} g**).
- **Final Held-Out Test Split Performance:**
  - **MAE:** **{test_metrics_best['mae']} grams**
  - **RMSE:** **{test_metrics_best['rmse']} grams**
  - **$R^2$ Score:** **{test_metrics_best['r2']}**

## 4. Comparison to Published Benchmark Baseline
- The published IEEE Access baseline (Sanatbyek et al., 2025) reports an MAE of **90.95 g** using a full multi-task YOLOv12 model with an integrated weight regression head trained on raw RGB imagery at 640×640.
- Our lightweight geometric model `{best_model_name}` achieves an MAE of **{test_metrics_best['mae']} g** on the test split using only 2D bounding-box geometry and class ID, without requiring GPU training or multi-gigabyte image downloads.

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
"""
    with open(os.path.join(BASE_DIR, 'REPORT.md'), 'w', encoding='utf-8') as f:
        f.write(report_content)

    print(f"\nAll artifacts successfully saved to {BASE_DIR}")
    print(f"Total time elapsed: {round(time.time() - start_time, 2)} seconds")

if __name__ == '__main__':
    main()
