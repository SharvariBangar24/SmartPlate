# Classification Dataset Audit Report (Stage 3A)

This report details the audit of the `Food-101` and `IndianFood` datasets to determine the optimal strategy for the Stage 3 food classification model.

## 1. Food-101 Audit
- **Total Usable Images**: 101,000 (0 corrupted)
- **Total Classes**: 101
- **Images per Class**: Exactly 1,000 (perfectly balanced)
- **Structure**: Pre-defined split exists (`train.txt` and `test.txt`).
  - Train images: 75,750 (750 per class)
  - Test images: 25,250 (250 per class)
- **Characteristics**: Large-scale, high diversity, predominantly Western/Global dishes (e.g., `pizza`, `hamburger`, `sushi`, `pho`), but includes some Indian/South Asian representation (`samosa`, `chicken_curry`).

## 2. IndianFood Audit
- **Total Usable Images**: 4,000 (0 corrupted)
- **Total Classes**: 80
- **Images per Class**: Exactly 50 (perfectly balanced internally)
- **Structure**: **NO PRE-DEFINED SPLIT**. The dataset is a flat directory of 80 folders inside `extracted/Indian Food Images/Indian Food Images`. There is no `train`/`val`/`test` structure.
- **Characteristics**: Extremely small scale (only 50 images per class). Highly specific regional Indian cuisine (`aloo_gobi`, `biryani`, `gulab_jamun`).

## 3. Cross-Dataset Comparison
### Exact Class-Name Overlaps
- **0 exact matches found.**

### Semantic Overlap Candidates (Manual Review)
While exact strings do not match, the concepts heavily overlap and could cause model confusion if treated as mutually exclusive classes:
- `chicken_curry` (Food-101) <---> `butter_chicken`, `chicken_tikka_masala`, `chicken_razala` (IndianFood)
- `fried_rice` (Food-101) <---> `biryani` (IndianFood)
- `pancakes` (Food-101) <---> `malapua` (IndianFood)
- `bread_pudding` (Food-101) <---> `double_ka_meetha` (IndianFood)
- `donuts` (Food-101) <---> `balushahi` or `gulab_jamun` (visual similarities)

### Duplicate Check
- An MD5 hash comparison across all 105,000 images found **0 exact file duplicates** between the two datasets.

## 4. Domain Analysis
- **Scale Discrepancy**: There is a massive **20:1 scale imbalance** between the datasets. Food-101 provides 1,000 images per class, whereas IndianFood only provides 50.
- **Preparation Gap**: Food-101 is fully prepared for machine learning (curated train/test splits). IndianFood is raw and un-split.
- **Coverage**: Neither dataset alone is sufficient for "Smart Plate". Food-101 lacks deep regional coverage, and IndianFood entirely lacks global staples.

## 5. Final Recommendation

**Recommended Strategy: Strategy C (Train on a carefully mapped unified dataset)**

**Justification & Required Next Steps:**
Training separate classifiers (Strategy D) is inefficient and complex to deploy (requires a routing model). Training on only one dataset (A or B) defeats the purpose of a versatile Smart Plate app. We must unify them, but we cannot simply concatenate the folders.

Before training, we **MUST** perform the following data engineering steps:
1. **Split IndianFood**: Programmatically shuffle and split the 4,000 IndianFood images into an 80/10/10 train/val/test structure.
2. **Address the 20:1 Imbalance**: If we train directly on the unified set, the model will heavily bias toward Food-101 classes. We must either heavily augment/oversample IndianFood, or undersample Food-101 during dataloading.
3. **Resolve Semantic Collisions**: We should manually merge or carefully distinguish the overlapping concepts (e.g., mapping `butter_chicken` and `chicken_tikka_masala` as sub-classes of `chicken_curry`, or explicitly leaving them separate and relying on the model to learn fine-grained differences).

*Audit complete. No training or modifications were performed.*
