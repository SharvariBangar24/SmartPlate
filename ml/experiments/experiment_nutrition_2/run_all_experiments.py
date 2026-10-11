"""
Master Orchestration Runner: Nutrition Experiments Suite
=========================================================
Executes all 7 experiments sequentially in a fully reproducible pipeline:
  1. Experiment 1: Baseline Validation
  2. Experiment 2: Database Improvements & Preparation Differentiations
  3. Experiment 3: Machine Learning Nutrition Prediction Models
  4. Experiment 4: Portion Estimation Regressor Benchmarking
  5. Experiment 5: Dataset Suitability Audit
  6. Experiment 6: Fair Comparison & Synthesis Charts
  7. Experiment 7: Full Pipeline Error Propagation & Real Image Validation
"""

import os
import sys
import time

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def main():
    start_total = time.time()
    print("=" * 80)
    print("SMART PLATE: COMPREHENSIVE NUTRITIONAL EXPERIMENTATION SUITE (EXPERIMENT 2)")
    print("=" * 80)

    # 1. Baseline Validation
    from experiment_1_validate_baseline import validate_baseline
    validate_baseline()

    # 2. Database Improvements
    from experiment_2_database_improvement import run_experiment_2
    run_experiment_2()

    # 3. Nutrition Prediction Models
    from experiment_3_nutrition_models import run_experiment_3
    run_experiment_3()

    # 4. Portion Estimation Benchmark
    from experiment_4_portion_models import run_experiment_4
    run_experiment_4()

    # 5. Dataset Suitability Audit
    from experiment_5_dataset_suitability import audit_datasets
    audit_datasets()

    # 6. Fair Comparison & Synthesis Charts
    from experiment_6_fair_comparison import run_experiment_6
    run_experiment_6()

    # 7. Full Pipeline Validation
    from experiment_7_full_pipeline_validation import run_experiment_7
    run_experiment_7()

    elapsed = time.time() - start_total
    print("\n" + "=" * 80)
    print(f"[SUCCESS] All 7 nutrition experiments executed successfully in {elapsed:.1f}s.")
    print("=" * 80)

if __name__ == "__main__":
    main()
