# FoodSeg103 YOLO Rare Class Analysis (Train Split)

Before advancing to data augmentation or model retraining, a granular analysis of the `FoodSeg103_YOLO` **train** split was performed. Validation data was strictly excluded from this analysis.

## 1. Overall Statistics (Train Split Only)
- **Total Training Bounding Boxes**: 28,272
- **Total Classes**: 103 
- *Note: `steamed bun` and `porridge` have 0 bounding boxes in the training split because they likely only existed in the validation set or were lost during mask-to-contour conversion due to extreme noise/artifacts.*

## 2. Rarity Group Classification

All 103 classes were sorted by their bounding box frequency and classified into 5 rarity tiers. 
*(For the full granular listing of all 103 classes, see the generated `foodseg_rare_class_analysis.csv`)*.

### A. Extremely Rare (<= 10 bounding boxes) — **9 Classes**
These classes are critically underrepresented and practically invisible to deep learning networks.
1. `steamed bun` (0 boxes)
2. `porridge` (0 boxes)
3. `spinach` (4 boxes)
4. `pudding` (5 boxes)
5. `ginger` (7 boxes)
6. `egg tart` (8 boxes)
7. `ham` (8 boxes)
8. `noodles` (9 boxes)
9. `cabbage` (10 boxes)

### B. Very Rare (11-25 bounding boxes) — **9 Classes**
These classes trigger severe overfitting because the network memorizes the very few images they appear in.
1. `rice` (11 boxes)
2. `pizza` (12 boxes)
3. `dumpling` (12 boxes)
4. `candied dates` (14 boxes)
5. `peanut` (16 boxes)
6. `yam` (16 boxes)
7. `strawberry` (19 boxes)
8. `broccoli` (22 boxes)
9. `pork` (25 boxes)

### C. Rare (26-50 bounding boxes) — **11 Classes**
These classes exist but usually yield `mAP@50 < 0.2` due to lack of diverse lighting/scale variation.
*(e.g., `tea` (29), `beef` (33), `cucumber` (33), `abalone` (33), `popcorn` (34), `soy` (38), `apricot` (39), `papaya` (39), `cashew` (43), `red beans` (44), `milk` (47))*

### D. Medium (51-100 bounding boxes) — **18 Classes**
These classes perform adequately but still fall below the network's potential.
*(e.g., `lemon`, `chicken`, `tofu`, `orange`, `onion`, `avocado`, `crab`, `walnut`, etc.)*

### E. Common (> 100 bounding boxes) — **56 Classes**
These classes dominate the dataset and the loss function.
*(e.g., `broccoli`, `bacon`, `potato`, `shrimp`, etc.)*

## 3. Multiple Rare Classes in Single Images

To optimize augmentation and prevent unnecessarily ballooning the dataset size, we checked for images containing *multiple* rare classes simultaneously. 

**Finding**: There are exactly **18 training images** that contain two or more classes from the "Rarest 30" list. 
Examples include:
- `354.txt` contains: `candied dates` & `red beans`
- `4474.txt` contains: `peanut` & `cashew`
- `4764.txt` contains: `pork`, `milk`, & `hami melon`
- `3036.txt` contains: `cucumber` & `abalone`

*Implication:* If we augment image `4764.txt`, we boost three rare classes simultaneously! Prioritizing these 18 images for offline augmentation provides maximum ROI.

## 4. Recommended Augmentation Strategy & Targets

To balance the dataset for a YOLO object detector, the standard threshold for a class to be considered "learnable" is roughly **150-200 bounding boxes** of diverse representation. 

Recommended offline augmentation targets (applying combinations of rotation, scale, color jitter, and mosaic):

| Rarity Group | Current Box Range | Recommended Augmentation Multiplier | Target Box Count per Class |
|---|---|---|---|
| **Extremely Rare** | 0 - 10 | **20x** (Generate 20 augmented copies of each image) | ~100-200 |
| **Very Rare** | 11 - 25 | **10x** (Generate 10 augmented copies of each image) | ~150-250 |
| **Rare** | 26 - 50 | **5x** (Generate 5 augmented copies of each image) | ~150-250 |
| **Medium** | 51 - 100 | **2x** (Generate 2 augmented copies of each image) | ~150-300 |
| **Common** | > 100 | **1x** (Do nothing, let YOLO's online augmentation handle it) | Unchanged |

### Pre-Augmentation Checklist (Pending)
1. **Zero-Box Classes**: For `steamed bun` and `porridge`, augmentation is mathematically impossible (0 x 20 = 0). We MUST manually scrape or port at least 5 images of these foods into the training split before augmenting, or explicitly drop them from the `data.yaml`.
2. **Isolate Rare Images**: Write a script to isolate the images belonging to the Extremely/Very Rare groups and apply the targeted multipliers.
