"""
Experiment 6: Fair Comparison & Cross-Model Synthesis
=====================================================
Compiles unified comparison tables and high-resolution comparison charts
across all completed experiments:
  1. Baseline Validation (Exp 1)
  2. Database Configurations (Exp 2)
  3. Alternative ML Nutrition Models (Exp 3)
  4. Portion Estimation Benchmark (Exp 4)
  5. Dataset Suitability Survey (Exp 5)

Generates:
  - ml/experiments/experiment_nutrition_2/results/exp6_fair_comparison.csv
  - ml/experiments/experiment_nutrition_2/results/exp6_fair_comparison.json
  - ml/experiments/experiment_nutrition_2/charts/nutrition_experiment_charts.png
"""

import os
import json
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
EXP_DIR = os.path.join(BASE_DIR, "ml", "experiments", "experiment_nutrition_2")
RESULTS_DIR = os.path.join(EXP_DIR, "results")
CHARTS_DIR = os.path.join(EXP_DIR, "charts")
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(CHARTS_DIR, exist_ok=True)

def run_experiment_6():
    print("=" * 70)
    print("EXPERIMENT 6: FAIR COMPARISON & MULTI-MODEL SYNTHESIS")
    print("=" * 70)

    # 1. Load results from Exp 2, 3, 4
    exp2_csv = os.path.join(RESULTS_DIR, "exp2_database_comparison.csv")
    exp3_csv = os.path.join(RESULTS_DIR, "exp3_models_comparison.csv")
    exp4_csv = os.path.join(RESULTS_DIR, "exp4_portion_comparison.csv")

    df_exp2 = pd.read_csv(exp2_csv) if os.path.exists(exp2_csv) else pd.DataFrame()
    df_exp3 = pd.read_csv(exp3_csv) if os.path.exists(exp3_csv) else pd.DataFrame()
    df_exp4 = pd.read_csv(exp4_csv) if os.path.exists(exp4_csv) else pd.DataFrame()

    print(f"Loaded results from Exp 2 ({len(df_exp2)} rows), Exp 3 ({len(df_exp3)} rows), Exp 4 ({len(df_exp4)} rows).")

    # 2. Build Unified Synthesis Table
    synthesis_rows = []

    # Baseline Phase A vs Improved Phase A
    for _, r in df_exp2.iterrows():
        cfg = r["configuration"]
        synthesis_rows.append({
            "task_domain": "Phase_A_Database_Level" if "Phase_A" in cfg else "Phase_B_Measured_Weight",
            "model_or_config": cfg,
            "sample_count": 54 if "Phase_A" in cfg else 160,
            "cal_mae": r["cal_mae"], "cal_mape": r["cal_mape"], "cal_pct20": r["cal_pct20"],
            "pro_mae": r["pro_mae"], "pro_mape": r["pro_mape"], "pro_pct20": r["pro_pct20"],
            "fat_mae": r["fat_mae"], "fat_mape": r["fat_mape"], "fat_pct20": r["fat_pct20"]
        })

    # ML Nutrition Models (Test Split N=65)
    for model_name, grp in df_exp3.groupby("model"):
        c_row = grp[grp["nutrient"] == "Calories"].iloc[0]
        p_row = grp[grp["nutrient"] == "Protein"].iloc[0]
        f_row = grp[grp["nutrient"] == "Fat"].iloc[0]
        synthesis_rows.append({
            "task_domain": "Phase_B_ML_Nutrition_Models",
            "model_or_config": f"ML_{model_name}",
            "sample_count": int(c_row["test_samples"]),
            "cal_mae": c_row["mae"], "cal_mape": c_row["mape"], "cal_pct20": c_row["pct_within_20"],
            "pro_mae": p_row["mae"], "pro_mape": p_row["mape"], "pro_pct20": p_row["pct_within_20"],
            "fat_mae": f_row["mae"], "fat_mape": f_row["mape"], "fat_pct20": f_row["pct_within_20"]
        })

    synth_df = pd.DataFrame(synthesis_rows)
    out_csv = os.path.join(RESULTS_DIR, "exp6_fair_comparison.csv")
    out_json = os.path.join(RESULTS_DIR, "exp6_fair_comparison.json")

    synth_df.to_csv(out_csv, index=False)
    with open(out_json, 'w', encoding='utf-8') as f:
        json.dump(synthesis_rows, f, indent=2)

    print(f"\n[OK] Saved unified synthesis table to {out_csv}")
    print(f"[OK] Saved unified synthesis JSON to {out_json}")

    # 3. Generate High-Resolution Publication Charts
    chart_path = os.path.join(CHARTS_DIR, "nutrition_experiment_charts.png")
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    plt.subplots_adjust(hspace=0.35, wspace=0.30)

    # Plot 1: Phase A Database Calibration Comparison
    ax1 = axes[0, 0]
    p_a_rows = df_exp2[df_exp2["configuration"].str.contains("Phase_A")]
    labels_a = ["Baseline DB", "Corrected DB", "Prep-Aware DB"]
    cal_maes_a = p_a_rows["cal_mae"].values
    pro_maes_a = p_a_rows["pro_mae"].values
    fat_maes_a = p_a_rows["fat_mae"].values
    x_a = np.arange(len(labels_a))
    w = 0.25
    ax1.bar(x_a - w, cal_maes_a, w, label="Calories (kcal)", color="#1f77b4")
    ax1.bar(x_a, pro_maes_a, w, label="Protein (g)", color="#2ca02c")
    ax1.bar(x_a + w, fat_maes_a, w, label="Fat (g)", color="#d62728")
    ax1.set_xticks(x_a)
    ax1.set_xticklabels(labels_a, fontsize=9.5)
    ax1.set_ylabel("MAE (kcal or grams)", fontsize=10)
    ax1.set_title("Exp 2: Phase A Database MAE by Configuration", fontsize=11, fontweight="bold")
    ax1.legend(loc="upper right", fontsize=8.5)
    ax1.grid(True, linestyle=":", alpha=0.6, axis="y")

    # Plot 2: Phase B Weighed Nutrition Calorie & Fat MAPE
    ax2 = axes[0, 1]
    p_b_rows = df_exp2[df_exp2["configuration"].str.contains("Phase_B")]
    labels_b = ["Baseline\nScaling", "Corrected\nDatabase", "Prep-Aware\nDatabase"]
    cal_mape_b = p_b_rows["cal_mape"].values
    fat_mape_b = p_b_rows["fat_mape"].values
    x_b = np.arange(len(labels_b))
    ax2.bar(x_b - w/2, cal_mape_b, w, label="Calorie MAPE (%)", color="#ff7f0e")
    ax2.bar(x_b + w/2, fat_mape_b, w, label="Fat MAPE (%)", color="#9467bd")
    ax2.set_xticks(x_b)
    ax2.set_xticklabels(labels_b, fontsize=9.5)
    ax2.set_ylabel("MAPE (%)", fontsize=10)
    ax2.set_title("Exp 2: Phase B Weighed Dish MAPE Reduction", fontsize=11, fontweight="bold")
    ax2.legend(loc="upper right", fontsize=8.5)
    ax2.grid(True, linestyle=":", alpha=0.6, axis="y")
    for rect in ax2.patches:
        h = rect.get_height()
        if h > 0:
            ax2.annotate(f"{h:.1f}%", xy=(rect.get_x() + rect.get_width()/2, h),
                         xytext=(0, 2), textcoords="offset points", ha="center", va="bottom", fontsize=8)

    # Plot 3: Portion Regressor Benchmark (Exp 4)
    ax3 = axes[1, 0]
    p_models = df_exp4["model"].tolist()
    test_maes = df_exp4["test_mae_g"].tolist()
    # Filter to main competitive models
    sel_idx = [i for i, m in enumerate(p_models) if m in ["Median_Baseline", "Class_Mean_Baseline", "Ridge_Regression", "Tuned_RandomForest", "Current_HistGradBoost", "Ensemble_HGB_RF"]]
    sel_models = [p_models[i].replace("_", "\n") for i in sel_idx]
    sel_maes = [test_maes[i] for i in sel_idx]
    bars3 = ax3.bar(range(len(sel_models)), sel_maes, color="#17becf", edgecolor="k", alpha=0.85)
    bars3[4].set_color("#2ca02c")  # Current HGB
    bars3[5].set_color("#1f77b4")  # Ensemble
    ax3.set_xticks(range(len(sel_models)))
    ax3.set_xticklabels(sel_models, fontsize=8)
    ax3.set_ylabel("Test MAE (grams)", fontsize=10)
    ax3.set_title("Exp 4: Portion Regressor Test MAE on FPB Split", fontsize=11, fontweight="bold")
    ax3.grid(True, linestyle=":", alpha=0.6, axis="y")
    for rect in bars3:
        h = rect.get_height()
        ax3.annotate(f"{h:.1f}g", xy=(rect.get_x() + rect.get_width()/2, h),
                     xytext=(0, 2), textcoords="offset points", ha="center", va="bottom", fontsize=8)

    # Plot 4: ML Nutrition Models vs Rule-Based (Exp 3)
    ax4 = axes[1, 1]
    m3_cal = df_exp3[df_exp3["nutrient"] == "Calories"]
    m3_names = [m.replace("_", "\n") for m in m3_cal["model"]]
    m3_maes = m3_cal["mae"].tolist()
    bars4 = ax4.bar(range(len(m3_names)), m3_maes, color="#e377c2", edgecolor="k", alpha=0.85)
    bars4[1].set_color("#2ca02c")  # Linear regression winner
    ax4.set_xticks(range(len(m3_names)))
    ax4.set_xticklabels(m3_names, fontsize=8.5)
    ax4.set_ylabel("Calorie MAE (kcal)", fontsize=10)
    ax4.set_title("Exp 3: ML Nutrition Models Calorie MAE (Held-Out Test)", fontsize=11, fontweight="bold")
    ax4.grid(True, linestyle=":", alpha=0.6, axis="y")
    for rect in bars4:
        h = rect.get_height()
        ax4.annotate(f"{h:.1f}", xy=(rect.get_x() + rect.get_width()/2, h),
                     xytext=(0, 2), textcoords="offset points", ha="center", va="bottom", fontsize=8.5)

    plt.suptitle("Smart Plate Nutrition Analysis Improvement: Unified Experimental Synthesis", fontsize=14, fontweight="bold", y=0.99)
    plt.tight_layout()
    plt.savefig(chart_path, dpi=300)
    plt.close()
    print(f"[OK] Saved unified comparison charts to {chart_path}")
    return synthesis_rows

if __name__ == "__main__":
    run_experiment_6()
