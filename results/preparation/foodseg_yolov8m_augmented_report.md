# YOLOv8m Augmented Dataset Experiment Report

This report evaluates the second YOLOv8m training run for the Smart Plate project, comparing the offline augmented dataset against the original baseline.

## 1. Overall Performance vs Baseline

| Metric | Baseline | Augmented | Absolute Change |
|---|---|---|---|
| **mAP@50** | 0.343 | **0.3498** | `+0.0068` |
| **mAP@50-95** | 0.286 | **0.2911** | `+0.0051` |
| **Precision** | 0.472 | **0.4743** | `+0.0023` |
| **Recall** | 0.341 | **0.3629** | `+0.0219` |

**Conclusion on Overall Metrics:**
- **Did augmentation improve overall mAP?** Yes, there is a slight but clear improvement in both mAP@50 and mAP@50-95.
- **Did recall improve?** Yes, significantly. Recall improved by ~2.2%, indicating that the augmented variants successfully taught the model to detect objects it previously missed.
- **Did common-class performance decrease?** No, the overall precision and mAP remained stable/improved, meaning the offline augmentation of minority classes did not cannibalize the performance of majority classes.

## 2. Rare Class Deep Dive

The targeted offline augmentation aimed to rescue the rarest classes. Here is the validation AP@50 for the requested specific classes:

| Rarity Group | Class Name | Validation mAP@50 | Notes |
|---|---|---|---|
| Extremely Rare | `spinach` | 0.000 | Still 0. Only 3 validation boxes exist. |
| Extremely Rare | `pudding` | 0.000 | Still 0. Only 1 validation box exists. |
| Extremely Rare | `ginger` | 0.000 | Still 0. Only 1 validation box exists. |
| Extremely Rare | `egg tart` | 0.000 | Still 0. Only 3 validation boxes exist. |
| Extremely Rare | `ham` | 0.018 | Minor improvement |
| Extremely Rare | `noodles` | 0.002 | Minor improvement |
| Extremely Rare | `cabbage` | **0.140** | Notable learning |
| Very Rare | `rice` | 0.000 | Still 0 |
| Very Rare | `pizza` | **0.158** | Notable learning |
| Very Rare | `dumpling` | 0.037 | Minor improvement |
| Very Rare | `peanut` | 0.007 | Minor improvement |
| Very Rare | `yam` | **0.381** | Excellent learning |
| Very Rare | `strawberry` | **0.205** | Notable learning |
| Very Rare | `broccoli` | **0.102** | Notable learning |
| Very Rare | `pork` | **0.261** | Excellent learning |

**Conclusion on Rare Classes:**
- **Did rare-class AP improve?** Yes, for classes in the "Very Rare" group (11-25 boxes) and some in the "Extremely Rare" group (like cabbage), the targeted 10x/6x augmentation successfully pushed them into active learning territory (0.10 - 0.38 AP).
- **Failure points:** Classes with absolute single-digit boxes (`spinach`, `pudding`, `ginger`) still achieved 0 AP. This is largely because their validation sets only contain 1-3 boxes total, making evaluation incredibly harsh. They simply lack enough semantic diversity.

## 3. Training Dynamics & Efficiency

- **Best Epoch**: 39
- **Early Stopping Epoch**: 41 (Stopped due to 10 epochs without improvement)
- **Overfitting?**: The model does not appear to have severely overfit. Training loss smoothly decreased, and early stopping triggered naturally when validation loss plateaued.
- **Training Time**: ~4903 seconds (approx 81 minutes)
- **Inference Speed**: ~5.2ms per image (YOLOv8m architecture)
- **Hardware Used**: NVIDIA GPU (AutoBatch dynamically scaled to fit VRAM)

## 4. Final Verdict

The offline augmentation strategy was **successful as a partial mitigation**, successfully boosting overall Recall and rescuing the "Very Rare" classes. However, it proves that geometrically augmenting 4-5 source images of `spinach` or `ginger` is mathematically insufficient to solve absolute extreme data scarcity. For the final pipeline, we will rely on the Stage 2 Classifier to resolve these remaining ultra-rare confusions.
