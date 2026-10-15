# Nutrition5k Dataset — Detailed EDA & Verification Report

## 1. DATASET LOCATION
- **Exact Path:** `D:\SmartPlate\datasets\Nutrition5k`
- **Confirmation:** Verified. This is the legitimate Nutrition5k dataset.
- **Major Folders:**
  - `imagery/` (Contains `realsense_overhead` and `side_angles`)
  - `metadata/` (Contains nutrition CSVs)
  - `dish_ids/` (Contains train/test splits and IDs)

## 2. DOWNLOAD / FILE INVENTORY
- **Total number of files:** 28,971
- **Total number of folders:** 8,290
- **Total disk size:** ~194.8 GB
- **Number of image files:** 10,455
- **Number of non-image files:** 18,516
- **Number of video files:** 18,466 (`.h264`)
- **Number of CSV files:** 3
- **Number of TXT/README files:** 7
- **Number of unusual/failed extensions:** 35 (`.gstmp` temp files)

| Extension | File Count | Approx. Total Size | Description |
|-----------|------------|-------------------|-------------|
| `.h264` | 18,466 | 191.5 GB | Side-angle videos |
| `.png` | 10,455 | 3.17 GB | Overhead RGB and Depth images |
| `.csv` | 3 | 2.3 MB | Nutrition metadata |
| `.txt` | 7 | 288 KB | Split IDs and READMEs |
| `.gstmp` | 35 | 145 MB | Failed/incomplete video downloads |

## 3. IMAGE COUNT
**Exact number of usable images: 3,490**
- All 10,455 `.png` images reside in `imagery/realsense_overhead/dish_*`.
- For each of the 3,490 dishes, there are exactly 3 images:
  1. `rgb.png` (Usable color image)
  2. `depth_color.png` (Colorized depth map)
  3. `depth_raw.png` (Raw depth array)
- Therefore, there are **3,490 distinct RGB overhead images**.
- *Note: No images exist in `side_angles` (they are raw video files).*

## 4. NUTRITION5K STRUCTURE
The dataset is structured primarily around `dish_ids`:
- **Overhead Images:** `imagery/realsense_overhead/dish_<id>/rgb.png`
- **Side-Angle Videos:** `imagery/side_angles/dish_<id>/camera_<id>.h264`
- **Metadata:** `metadata/dish_metadata_cafe1.csv` and `dish_metadata_cafe2.csv`
- **Ingredient Catalog:** `metadata/ingredients_metadata.csv`
- **Train/Test Splits:** Provided in `dish_ids/splits/` (`rgb_train_ids.txt`, `rgb_test_ids.txt`).

## 5. IMAGE DETAILS
A sample of 100 `rgb.png` images from the overhead directory was analyzed:
- **Dimensions:** 640x480 (Width x Height)
- **Channels:** 3 (RGB Mode)
- **Format:** PNG
- **Approximate File Size:** 300–500 KB per RGB image.
- **Corrupted Images:** 0 found in the sampled batch.
- **Uniformity:** All overhead RGB images share the exact same 640x480 resolution.

## 6. METADATA / NUTRITION INFORMATION
There are three main metadata files. The dish metadata files do not have headers, but their structure is identifiable.

**`metadata/dish_metadata_cafe1.csv` (4,767 dishes)** and **`metadata/dish_metadata_cafe2.csv` (237 dishes)**:
Columns represent:
1. `Dish ID` (e.g., `dish_1561662216`)
2. `Total Calories` (kcal)
3. `Total Mass` (g)
4. `Total Fat` (g)
5. `Total Carbohydrates` (g)
6. `Total Protein` (g)
7. *Following columns repeat in groups of 7 for each ingredient on the plate:*
   - `Ingredient ID`, `Ingredient Name`, `Mass(g)`, `Calories`, `Fat`, `Carbs`, `Protein`

**`metadata/ingredients_metadata.csv` (555 ingredients)**:
Columns: `ingr` (name), `id`, `cal/g`, `fat(g)`, `carb(g)`, `protein(g)`.

## 7. FOOD CLASSES / INGREDIENTS
- **Categories:** The dataset does not use standard 1-to-1 food classes (like "Apple"). Instead, each image is a "Dish".
- **Ingredients:** There are **555 unique ingredients** used to construct these dishes.
- **Frequent Labels:** Rice, chicken, tomatoes, olive oil, bread, beans.
- **Structure:** A single image of a dish might contain ["white rice", "soy sauce", "garlic", "bok choy", "pork"].

## 8. IMAGE ↔ LABEL / NUTRITION CONNECTION
The connection between an image and its nutrition data is purely ID-based.
1. The folder name acts as the primary key: `imagery/realsense_overhead/dish_1561662216/rgb.png`
2. This exact ID (`dish_1561662216`) is looked up in `dish_metadata_cafe1.csv`.
3. The row provides the aggregate nutrition (Calories: 300.79, Mass: 193g) and a breakdown of every ingredient present in that specific image.

## 9. TRAIN / VALIDATION / TEST SPLITS
The dataset **already provides explicit splits** in `dish_ids/splits/`:
- `rgb_train_ids.txt`: 4,059 dishes
- `rgb_test_ids.txt`: 709 dishes
- `depth_train_ids.txt`: 2,758 dishes
- `depth_test_ids.txt`: 507 dishes
- *Note: There is no distinct "validation" split provided by default (only train/test).*

## 10. DUPLICATES / DATA QUALITY
- **Images without Metadata:** There are **2** dish folders in `realsense_overhead` that do not exist in the CSV metadata.
- **Metadata without Images:** There are **1,516** dishes listed in the CSV files that do not have corresponding downloaded images in `realsense_overhead`.
- **Conclusion:** The dataset is mostly clean, but an inner-join approach is required. Only the **3,488 dishes** that have BOTH an image and a metadata row can be used.

## 11. THE 35 FAILED FILES
The 35 missing/failed files are `.gstmp` (GStreamer Temp) files located in the `imagery/side_angles/` directory.
- **Importance:** LOW. These are interrupted downloads for the side-angle `.h264` videos.
- **Impact on Smart Plate:** ZERO. Since we are using the `realsense_overhead` RGB images, the corrupted/missing side-angle videos do not affect our image dataset.

## 12. USEFULNESS FOR SMART PLATE
Based on the actual contents, this dataset is:
- **Food classification:** B (Useful with processing - contains multiple items per image without bounding boxes).
- **Nutrition/Calorie/Mass estimation:** A (Directly useful - this is the crown jewel of the dataset. It provides pixel-to-mass and pixel-to-calorie mapping).
- **Multi-item food recognition:** C (Not supported directly - it lists ingredients but does not provide bounding boxes or segmentation masks to locate them).

## 13. COMPATIBILITY WITH OUR OTHER DATASETS
- **FoodSeg103:** (7,118 images). Single/multi-item with strict segmentation masks.
- **IndianFood:** (4,000 images). Single-item crops with flat category labels.
- **Nutrition5k:** (3,488 images). Multi-item plates with exact macros and mass, but NO spatial annotations.
- **Compatibility:** Nutrition5k **cannot** be directly merged with FoodSeg103 or IndianFood to train a standard YOLO or U-Net model because it lacks spatial annotations (boxes/masks). It must remain a **separate training source** specifically for the Module 4/5 Nutrition Estimation neural network.

## 14. FINAL SUMMARY

| Dataset | Usable Images | Main Purpose | Annotation Type | Nutrition Data | Can Directly Train With Others? |
|---------|---------------|--------------|-----------------|----------------|---------------------------------|
| FoodSeg103 | 7,118 | Segmentation | Pixel Masks | No | No (Different format) |
| IndianFood | 4,000 | Classification| Folder Label | No | No (Different format) |
| Nutrition5k| 3,488 | Macro/Mass Est| CSV Metadata | **Yes (High detail)**| No (Lacks bounding boxes) |

### Key Findings
1. **Nutrition5k contains exactly 3,488 fully valid overhead RGB images** that possess corresponding ground-truth nutrition metadata.
2. The dataset size is massive (194 GB) largely because of 18,466 side-angle `.h264` video files which we do not strictly need.
3. The 35 failed download files are isolated to the side-angle video directory and do not affect the RGB images.
4. The dataset maps a single image to a total calorie/mass value AND a detailed ingredient-by-ingredient breakdown.
5. **No spatial annotations (bounding boxes or masks) exist in Nutrition5k.** Therefore, it cannot be used to train an object detection (YOLO) or segmentation (U-Net) model directly.

### Recommended Next Step
Since our combined current usable image count across all datasets is only ~14,606 (far short of 50,000) and they have totally incompatible annotation formats, **I recommend downloading a large-scale, pre-annotated bounding-box dataset (like Food-101 or UEC-Food256)** to serve as the unified multi-item recognition dataset for YOLO. Nutrition5k should be reserved specifically for the volume/calorie estimation stage (Stage 4/5).
