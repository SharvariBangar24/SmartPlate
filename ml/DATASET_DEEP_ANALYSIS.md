# Deep Dataset Technical Analysis

## A. IMAGE STRUCTURE
- **UEC FOOD-256**: ~31,395 images. Predominantly single-food center-focused images, though multi-food exists. Exact count of multi-item images: *Cannot determine without downloading/processing the dataset.*
- **FoodInsSeg**: 7,118 images. Exclusively complex multi-item (average 16-17 ingredient instances per image).
- **Indian Food Object Detection**: 1,211 images. Mix of single and multi-food images (Thali style). Exact multi-item breakdown: *Cannot determine without downloading/processing the dataset.*
- **SOHL Multi-Dish**: 377 images. Heavily multi-food focused.

## B. ANNOTATION STRUCTURE
- **UEC FOOD-256**: Bounding boxes. Stored in a custom `bb_info.txt` file format inside each of the 256 folders. Identifies categories via folder hierarchy.
- **FoodInsSeg**: Instance segmentation masks (polygons). Stored in COCO JSON format. Identifies 103 ingredient categories.
- **Indian Food Object Detection**: Bounding boxes. YOLO `.txt` format. Identifies 14 Indian dishes.
- **SOHL Multi-Dish**: Bounding boxes. YOLO `.txt` format. Identifies 16 general Indian dish types.

## C. CLASS INFORMATION
The complete class list has been documented separately in `DATASET_CLASS_LISTS.md`.
- **Useful for Smart Plate**: The 14 classes from IFOD and 16 classes from SOHL are highly useful. FoodInsSeg's 103 ingredients are highly useful for Phase 2 (portion/nutrition). UEC's 256 classes are largely *useless* as final targets (Japanese focus), but useful as generic objects.

## D. DATA QUALITY
- **UEC FOOD-256**: High quality research dataset. No missing annotations expected, but requires rigorous format conversion.
- **FoodInsSeg**: Extremely high quality mask annotations. Peer-reviewed.
- **Indian Food Object Detection**: Unknown until downloaded, but reported mAP (47%) suggests standard community quality.
- **SOHL Multi-Dish**: Inspected locally. 0 missing labels, 0 empty labels, 0 invalid coordinates. Extremely clean YOLO format.

## E. MULTI-ITEM ANALYSIS
- **UEC FOOD-256**: *Cannot determine without downloading/processing the dataset.*
- **FoodInsSeg**: *Cannot determine without downloading/processing the dataset.* (However, dataset averages 16.59 masks per image, meaning near 100% multi-item).
- **Indian Food Object Detection**: *Cannot determine without downloading/processing the dataset.*
- **SOHL Multi-Dish** (Locally computed):
  - 1 item: 16.4% (62)
  - 2 items: 22.3% (84)
  - 3 items: 26.8% (101)
  - 4+ items: 34.5% (130)

## F. SMART PLATE RELEVANCE (1-5 Rating)

| Dataset | Multi-item | Indian Food | Ann. Quality | YOLO Comp. | Pretrain | Finetune |
|---------|------------|-------------|--------------|------------|----------|----------|
| UEC FOOD-256 | 3 | 1 | 4 | 2 | 5 | 1 |
| FoodInsSeg | 5 | 3 | 5 | 2 | 3 | 2 |
| Indian Food OD | 4 | 5 | 4 | 5 | 2 | 5 |
| SOHL Multi-Dish | 5 | 5 | 5 | 5 | 1 | 5 |

## G. DATASET ROLE
- **UEC FOOD-256**: Pretraining only.
- **FoodInsSeg**: Segmentation / Portion estimation phase.
- **Indian Food Object Detection**: Indian-domain fine-tuning, validation/testing.
- **SOHL Multi-Dish**: Indian-domain fine-tuning, validation/testing.

## H. 20K–30K FEASIBILITY

**Scenario A: UEC useful images + Indian detection datasets**
RAW IMAGES: ~32,983 (31,395 + 1,211 + 377)
→ potentially usable images (filtering UEC for multi-item): ~8,000
→ expected after quality filtering: ~7,500
→ expected after duplicate removal: ~7,000

**Scenario B: UEC + FoodInsSeg + Indian datasets**
RAW IMAGES: ~40,101
→ potentially usable images: ~15,000
→ expected after quality filtering: ~14,000
→ expected after duplicate removal: ~13,500

**Scenario C: Only datasets suitable for actual object detection (No UEC, No Masks)**
RAW IMAGES: ~1,588
→ potentially usable images: ~1,588
→ expected after duplicate removal: ~1,500

**Conclusion on Feasibility:**
We **CANNOT** construct a scientifically valid 20k-30k image dataset of *specifically multi-item Indian food bounding boxes* using only these four sources. UEC provides the sheer volume, but is predominantly Japanese and single-item.

## I. IMPORTANT UEC ANALYSIS
**Option 3 is strictly recommended**: Use UEC *only* for pretraining, and keep the final Smart Plate classes entirely Indian-food focused.
**Reason:** Mapping UEC's 256 Japanese classes to our 16 Indian classes is impossible without destroying data integrity (e.g., forcing Sushi to become "snack_item"). Pretraining on UEC teaches the model spatial features ("what a plate looks like", "what a table looks like"), allowing the final fine-tuning on our smaller Indian datasets to converge rapidly and accurately without hallucinating Japanese dishes.

## J. FOODINSEG ANALYSIS
FoodInsSeg must be treated as a **portion estimation source** and a **segmentation source**, *not* as an object detection source for Phase 1.
Segmentation masks provide 2D pixel area, which is mathematically required for estimating volume (Area × Estimated Depth). However, masks do not inherently solve portion estimation; they are simply the prerequisite step before applying density/volume heuristics.

---

## Dataset Decision

**PRIMARY PRETRAINING DATASET:** UEC FOOD-256 (Treated strictly as a generic feature extractor).
**PRIMARY INDIAN DATASET:** Indian Food Object Detection (1,211 images).
**SECONDARY DATASET:** SOHL Multi-Dish (377 images).
**SEGMENTATION DATASET:** FoodInsSeg (Held for Phase 2).
**VALIDATION DATASET:** A 15% holdout split derived from combining the two Indian datasets.

## Realistic Final Dataset Size
**~1,500 highly accurate, multi-item Indian food images.**
*(Note: Achieving the 20k–30k goal requires either downloading a massive specialized dataset like 'Nutrition5k' or writing automated scraping scripts for Indian food APIs. The data present here does not support a 20k+ size without severe quality dilution).*

## Critical Findings
1. **The 30k Goal is Unreachable with Current Sources:** Filtering for multi-item Indian food strictly reduces the pool to ~1,500 images.
2. **UEC Must Not Be Merged:** Merging UEC's 256 classes with our Indian classes will cause catastrophic class confusion. It must be used purely for transferring spatial weights.
3. **No Masks = No Volume:** Datasets 1, 3, and 4 only provide bounding boxes, which are insufficient for accurate nutrient estimation. FoodInsSeg (or SAM) is mandatory for Phase 2.
4. **Class Taxonomy Clashes:** SOHL uses generic names (`rice_dish`), while IFOD uses specific names (`Biryani`). These must be manually harmonized before any code is run.
5. **No Native Splits:** None of the Indian datasets have guaranteed, pre-defined Train/Val/Test splits locally. A strict script must be written to prevent data leakage.

## Next Step
Define the **Final Smart Plate Class Taxonomy** by mapping the generic categories (from SOHL) to the specific categories (from IFOD) into a single, unified list of ~15 target classes, so the dataset merge strategy can be established.
