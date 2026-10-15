# Pre-Training Preparation Report — Smart Plate

================================================

## 1. IndianFood Split Results

A deterministic 80/10/10 train/validation/test split was successfully generated for the `IndianFood` dataset using a fixed random seed (`42`). The dataset was completely preserved; the splits were written to a mapping file (`D:\SmartPlate\results\preparation\indianfood_splits.json`).

- **Total Classes**: 80
- **Total Images**: 4,000
- **Stratification Strategy**: Exact Class Balancing
- **Train Split**: 3,200 images (exactly 40 images per class)
- **Validation Split**: 400 images (exactly 5 images per class)
- **Test Split**: 400 images (exactly 5 images per class)

## 2. Nutrition5k Mapping Results

The `dish_metadata_cafe1.csv` and `dish_metadata_cafe2.csv` metadata files were verified against the downloaded `imagery/realsense_overhead/` directories.

- **Total Metadata Dishes (Rows in CSVs)**: 5,006
- **Total Downloaded Imagery Dishes (Folders with `rgb.png`)**: 3,485
- **Valid Image ↔ Metadata Pairs**: **3,485**
- **Missing Metadata for Images**: 0 (Every downloaded image has metadata)
- **Missing Images for Metadata**: 1,521 (We are missing 1.5k images described in the CSV, likely due to a partial dataset download, but this is acceptable as we have 3,485 clean pairs).
- **Invalid Rows / Duplicate IDs**: 0

*Conclusion: We have 3,485 perfectly clean, ready-to-train Nutrition5k samples.*

## 3. Duplicate & Leakage Findings

An MD5 hashing audit was performed between the **IndianFood** and **Food-101** datasets to check for exact data leakage (i.e., images scraped from the same source appearing in both datasets).

- **IndianFood files hashed**: 3,976 (usable files)
- **Food-101 files hashed**: 101,000
- **Exact duplicate pairs found**: **0**

*Conclusion: There is no exact file-level data leakage between Food-101 and IndianFood. The datasets are disjoint and safe to use together.*

## 4. Class-Overlap Findings

A comparison was made between the 101 classes of Food-101, the 80 classes of IndianFood, and the known broad categories of FoodSeg103.

### Exact Matches
- Surprisingly, there are **ZERO exact class name matches** between IndianFood and Food-101. (For example, Food-101 contains `samosa`, but IndianFood does not include a `samosa` class). 

### Similar Foods (Potential Confusion Risks)
- **Food-101**: `fried_rice`
- **IndianFood**: `biryani`, `poha`
- **FoodSeg103**: `rice`, `fried rice`

- **Food-101**: `garlic_bread`
- **IndianFood**: `naan`, `chapati`, `bhatura`, `misi_roti`
- **FoodSeg103**: `bread`

### Broad vs. Specific Relationships
- **Poultry**: Food-101 has the broad `chicken_curry`. IndianFood specifies `butter_chicken`, `chicken_tikka`, `chicken_tikka_masala`, `kadai_paneer`. FoodSeg103 just has `chicken`.
- **Desserts**: Food-101 has `ice_cream`, `carrot_cake`. IndianFood has `gulab_jamun`, `jalebi`, `rasgulla`.

*Conclusion: The datasets are highly complementary. Food-101 provides broad international foods, IndianFood provides highly granular regional dishes, and FoodSeg103 provides foundational object bounds (e.g., "bread", "chicken").*

## 5. Problems Discovered

1. **IndianFood Image Count**: Hashing found 3,976 valid image files, not 4,000. 24 files are either corrupted or not valid image extensions.
2. **Missing Nutrition5k Images**: 1,521 metadata rows do not have downloaded imagery. We can only use the 3,485 pairs.
3. **No Detection Annotations**: We still lack YOLO bounding box annotations (`.txt`), which is the final roadblock before training Stage 1.

## 6. Recommended Next Step

The absolute next required step is to **convert the FoodSeg103 segmentation masks into YOLO bounding boxes**. 
Since we plan to use YOLOv8 or a similar detector to isolate food items on a plate before passing them to the IndianFood/Food-101 classifier, we cannot train until those bounding boxes exist.

I recommend running a conversion script over `datasets/FoodSeg103/masks` to calculate the bounding box `[x_center, y_center, width, height]` for each distinct class ID blob, and saving them to a `labels/` directory.
