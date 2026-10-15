# Rare Class Augmentation Candidates

This report identifies specific training images in `FoodSeg103_YOLO_101` that require offline augmentation to rectify the dataset's extreme class imbalance. The analysis strictly focuses on the training split.

## 1. Train Split Distribution Summary
- **Total Valid Classes**: 103 (0 boxes classes removed)
- **Extremely Rare (<= 10 boxes)**: 7 classes
- **Very Rare (11-25 boxes)**: 9 classes
- **Rare (26-50 boxes)**: 11 classes

## 2. Multi-Rare Image Identification
To maximize the efficiency of offline augmentation, images containing *multiple* rare classes simultaneously should be prioritized. Enhancing one such image boosts several minority classes at once.

**Top Priority Multi-Rare Images:**
- `4930.jpg`: Contains `abalone`, `noodles`, `pizza`
- `4506.jpg`: Contains `apricot`, `hami melon`
- `1699.jpg`: Contains `milk`, `egg tart`
- `4717.jpg`: Contains `cashew`, `popcorn`
- `1063.jpg`: Contains `pudding`, `papaya`
- `4764.jpg`: Contains `pork`, `milk`
- `4798.jpg`: Contains `pork`, `milk`
- `354.jpg`: Contains `red beans`, `candied dates`
- `2720.jpg`: Contains `abalone`, `red beans`
- `4921.jpg`: Contains `red beans`, `beef`
- `4474.jpg`: Contains `cashew`, `peanut`
- `4558.jpg`: Contains `cashew`, `ham`
- `4857.jpg`: Contains `cashew`, `peanut`
- `3036.jpg`: Contains `abalone`, `cucumber`
- `4787.jpg`: Contains `ham`, `spinach`
- `4886.jpg`: Contains `dumpling`, `beef`
- `1454.jpg`: Contains `abalone`, `pizza`
- `4702.jpg`: Contains `noodles`, `pizza`

## 3. Class Vulnerability Details

### Extremely Rare
- `spinach`: 4 boxes (1 image: `4787.jpg`)
- `pudding`: 5 boxes (4 images: `1780.jpg`, `412.jpg`, `1063.jpg`, `2432.jpg`)
- `ginger`: 7 boxes (5 images)
- `egg tart`: 8 boxes (3 images)
- `ham`: 8 boxes (3 images)
- `noodles`: 9 boxes (6 images)
- `cabbage`: 10 boxes (9 images)

### Very Rare
- `rice`: 11 boxes
- `pizza`: 12 boxes
- `dumpling`: 12 boxes
- `candied dates`: 14 boxes
- `peanut`: 16 boxes
- `yam`: 16 boxes
- `strawberry`: 19 boxes
- `broccoli`: 22 boxes
- `pork`: 25 boxes

## 4. Augmentation Recommendations

**Priority 1: Multi-Rare Images**
- Heavily augment (15x-20x via rotation, scaling, mosaic) the 18 multi-rare images listed above. This is the most computationally efficient way to lift the baseline for `spinach`, `noodles`, `pizza`, `ham`, and `cashew`.

**Priority 2: Extremely Rare Singles**
- For classes like `ginger` and `cabbage` that lack multi-rare overlaps, extract their individual training images and apply 20x heavy augmentation to brute-force their representation up to ~150-200 bounding boxes.

**Priority 3: Very Rare & Rare Singles**
- Apply a moderate 5x-10x multiplier to the images in these groups to ensure they comfortably cross the YOLO learning threshold.

**Priority 4: Common Classes**
- Do absolutely nothing. Rely on YOLO's native online augmentation (Mosaic/MixUp) during training to handle variation for `bacon`, `broccoli`, etc.

*Detailed class breakdowns and multi-rare candidates are fully mapped in the generated CSV files.*
