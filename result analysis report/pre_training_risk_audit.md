# PRE-TRAINING RISK & DATASET AUDIT — SMART PLATE

================================================

## PART 1 — COMPLETE DATASET INVENTORY

| Dataset | Total Files | Folders | Usable Images | Non-Image | Extensions | Image Res | Channels | Missing/Corrupt | Metadata/Annotation | Splits |
|---|---|---|---|---|---|---|---|---|---|---|
| **FoodSeg103** | 14,237 | 4 | **7,118** | 7,119 | `.jpg`, `.png`, `.py` | Variable | RGB/RGBA | 0 | 7,118 PNG Masks | Train/Val |
| **IndianFood** | 4,002 | 80 | **4,000** | 2 | `.jpg`, `.zip` | Variable | RGB | 0 | Folder Labels | None |
| **Nutrition5k** | 28,971 | 8,290 | **3,490** (RGB) | 18,516 | `.png`, `.h264`, `.csv`, `.gstmp` | 640x480 | RGB | 35 `.gstmp` (videos) | CSV Metadata | Train/Test |
| **Food-101** | 101,008 | 103 | **101,000** | 8 | `.jpg`, `.txt`, `.json` | Variable | RGB | 0 | Folder Labels, `.txt` splits | Train/Test |
| **Nutrition** | 35 | 1 | **0** | 35 | `.csv`, `.png` (plots) | N/A | N/A | 0 | Tabular/Plots | None |
| **TOTAL** | **148,253** | **8,478** | **115,608** | **25,680** | | | | | | |

*Note: Nutrition5k contains depth maps and side-angle `.h264` videos, which are not included in the usable RGB image count above.*

================================================

## PART 2 — DATASET-SPECIFIC STRUCTURE

### 1. FoodSeg103
- **Type**: Semantic Segmentation.
- **Sample Represents**: An image containing one or multiple food items.
- **Labels**: 104 pixel-wise classes (103 food types + 1 background).
- **Ground Truth**: PNG masks mapping each pixel to a class ID.
- **Can Learn**: Boundary delineation, shape recognition, and multi-item isolation.
- **CANNOT be used for**: Nutrition regression, direct YOLO bounding-box detection (without mask-to-box conversion).

### 2. IndianFood
- **Type**: Classification.
- **Sample Represents**: A single cropped image of an Indian dish.
- **Labels**: 80 specific Indian dish categories.
- **Ground Truth**: Directory structure (folder name = class).
- **Can Learn**: Visual identification of 80 distinct Indian foods.
- **CANNOT be used for**: Segmentation, object detection, or nutrition estimation.

### 3. Food-101
- **Type**: Classification.
- **Sample Represents**: A single dish/food item in varied environments.
- **Labels**: 101 international food categories.
- **Ground Truth**: Directory structure and JSON/TXT splits.
- **Can Learn**: Robust, general food classification across 101 common dishes.
- **CANNOT be used for**: Segmentation, multi-item detection, nutrition estimation.

### 4. Nutrition5k
- **Type**: Nutrition Regression / Image-to-Mass.
- **Sample Represents**: An overhead plate with one or more ingredients.
- **Labels**: Dish IDs mapped to macronutrient vectors.
- **Ground Truth**: CSV rows detailing total calories, mass, fat, carbs, protein, and ingredient breakdowns.
- **Can Learn**: Mapping visual volume/pixels directly to physical mass and calories.
- **CANNOT be used for**: Object detection/segmentation (no spatial annotations exist).

### 5. Nutrition
- **Type**: Tabular/Metadata.
- **Sample Represents**: Distribution histograms and correlation heatmaps.
- **CANNOT be used for**: Any visual training. Useful only as reference material.

================================================

## PART 3 — LABEL / ANNOTATION AUDIT

### Classification (Food-101 & IndianFood)
- **Food-101**: 101 classes. 1,000 images per class exactly (750 train / 250 test). Total = 101,000. Perfect balance.
- **IndianFood**: 80 classes. 50 images per class exactly. Total = 4,000. Perfect balance, but extremely small sample size per class.

### Segmentation (FoodSeg103)
- **Image-mask pairing**: 1:1 mapping (7,118 images = 7,118 masks).
- **Classes**: 104 classes.
- **Issues**: High likelihood of class imbalance at the pixel level (e.g., "rice" pixels vastly outnumber "avocado" pixels).

### Detection
- **NO DATASET HAS BOUNDING BOXES.** 
- There are no YOLO `.txt` or COCO `.json` files providing `[x, y, w, h]` annotations in the entire `datasets/` directory.

### Nutrition5k
- **Fields**: Total Calories, Total Mass, Fat, Carbs, Protein, Ingredient IDs, Ingredient mass/macros.
- **Matching**: `imagery/realsense_overhead/dish_ID/rgb.png` -> `metadata/dish_metadata_cafe1.csv` row where ID = `dish_ID`.
- **Missing Images**: 1,516 dishes listed in the metadata do not have downloaded imagery.
- **Missing Metadata**: 2 images exist without metadata rows.

================================================

## PART 4 — CLASS IMBALANCE

- **Food-101**: perfectly balanced at the image level (1,000 per class).
- **IndianFood**: perfectly balanced at the image level (50 per class).
- **FoodSeg103**: Highly imbalanced at the pixel level. Background class dominates. Common foods (rice, bread, meat) dominate rare foods.
- **Nutrition5k**: Imbalanced ingredient distributions (many dishes have rice/chicken, very few have exotic ingredients).

**Recommendations**:
- **Do NOT** oversample or augment the validation/test sets. They must reflect real-world distributions.
- For Segmentation (FoodSeg103): Use **Class Weighting** (e.g., in Focal Loss or CrossEntropy) to penalize background and common classes less, and rare classes more.
- For IndianFood: 50 images per class is very low. Use **heavy, controlled augmentation** to prevent overfitting.
- For Food-101: Leave unchanged.

================================================

## PART 5 — CLASS NAME OVERLAP

There are 101 classes in Food-101, 80 in IndianFood, and 103 in FoodSeg103.

| Food-101 (General) | IndianFood (Specific) | FoodSeg103 (Ingredients) | Recommendation |
|---|---|---|---|
| `samosa` | `samosa` | - | A. Exact Match. |
| `chicken_curry` | `butter_chicken`, `chicken_tikka` | `chicken` | F. Keep separate. IndianFood provides needed granularity. |
| `fried_rice` | `biryani`, `poha` | `rice`, `fried rice` | G. Segment as "Rice/Grain", Classify specifically. |
| `garlic_bread` | `naan`, `chapati` | `bread` | G. Segment as "Bread", Classify specifically. |

**Strategy**: Do NOT merge the label sets. Use FoodSeg103 strictly to identify "food blobs" (e.g., Meat, Bread, Rice). Then crop the blob and pass it to a classification model trained jointly on Food-101 + IndianFood to say "This Meat blob is a Pork Chop" or "This Bread blob is Naan".

================================================

## PART 6 — DUPLICATES & DATA LEAKAGE

**Leakage Risks**:
- **Dataset overlap**: Food-101 contains a `samosa` class. IndianFood contains a `samosa` class. If an image was scraped from the same internet source and appears in Food-101's train set and IndianFood's test set, evaluating a joint model will yield artificially inflated accuracy.
- **Augmentation leakage**: If an augmented image is placed in the validation set while the original is in the train set, the model just memorizes the image.

**Recommended Safe Strategy**:
- Split datasets *independently* BEFORE any processing or augmentation.
- For IndianFood (no split provided): Create a deterministic stratified split (e.g., 80% train, 10% val, 10% test) using a fixed random seed.
- For Food-101 and FoodSeg103: Strictly use their provided Train/Test splits. Do not mix them.
- Do not evaluate IndianFood test data on a model trained only on Food-101, unless specifically testing zero-shot/domain transfer.

================================================

## PART 7 — DATASET COMPATIBILITY

| Combination | Status | Why |
|---|---|---|
| FoodSeg103 + IndianFood | **Not Compatible (Directly)** | Segmentation vs Classification. Different annotation formats. |
| Food-101 + IndianFood | **Compatible** | Both are flat folder-based classification datasets. |
| Nutrition5k + Food-101 | **Not Compatible** | Nutrition5k requires mapping visual volume to exact macros. Food-101 has no volume/depth or macro data. |
| Nutrition5k + FoodSeg103 | **Partially Compatible** | Could potentially run FoodSeg103 segmentation on Nutrition5k images to isolate ingredients before running mass estimation. |

================================================

## PART 8 — IMAGE DOMAIN ANALYSIS

- **Food-101 & IndianFood**: "In-the-wild" internet photos. Varied lighting, angles, restaurant/home settings, often close-ups.
- **Nutrition5k**: Laboratory/Cafe setting. Fixed 640x480 resolution. Fixed overhead camera angle. Fixed lighting. Black cafeteria trays.
- **FoodSeg103**: Varied internet images, but heavily annotated.

**Domain Shift Problem**: A model trained to estimate calories purely on Nutrition5k will learn to expect a fixed distance overhead shot on a specific tray. If you feed it a close-up smartphone photo from IndianFood, it will catastrophically fail at estimating mass because the scale/perspective is completely different.

================================================

## PART 9 — REAL SMART PLATE RISKS

The real Smart Plate system must handle:
1. **Multiple overlapping foods**: ✅ Handled by FoodSeg103 (segmentation).
2. **Indian dishes**: ✅ Handled by IndianFood.
3. **General foods**: ✅ Handled by Food-101.
4. **Nutrition estimation**: ⚠️ Handled by Nutrition5k, BUT suffers from massive domain shift (fixed camera vs smartphone).
5. **Detection (Bounding Boxes)**: ❌ **CRITICAL FAILURE.** We do not have a bounding box dataset to train a YOLO detector.
6. **Oil/Fat NIR**: ❌ Not present yet.

================================================

## PART 10 — TRAINING RISKS

1. **PROBLEM: Detection Missing**
   → *Why:* No bounding box annotations exist.
   → *Detect:* Trying to train YOLO will immediately throw errors for missing `.txt` label files.
   → *Prevent:* Must run a script to convert FoodSeg103 PNG masks into YOLO bounding boxes before Stage 1 training.

2. **PROBLEM: Extreme Domain Shift in Nutrition**
   → *Why:* Nutrition5k is highly standardized (fixed camera/lighting).
   → *Detect:* Calorie estimation model gets 95% accuracy on Nutrition5k test set, but outputs random numbers on cell phone photos.
   → *Prevent:* Train the nutrition regression model using heavy scale/perspective/lighting augmentations to force it to look at food textures, not the plate borders.

3. **PROBLEM: Overfitting IndianFood**
   → *Why:* Only 50 images per class. Deep networks like ResNet50 will memorize 50 images in a few epochs.
   → *Detect:* Training loss drops to 0, validation loss spikes (classic U-curve).
   → *Prevent:* Early stopping, high dropout, aggressive data augmentation, and heavy weight decay.

4. **PROBLEM: Class Confusion**
   → *Why:* Food-101 `fried_rice` vs IndianFood `biryani` are visually nearly identical.
   → *Detect:* High off-diagonal values in the Confusion Matrix between these two classes.
   → *Prevent:* Group them under a parent class, or use a hierarchical classifier.

================================================

## PART 11 — MODEL STRATEGY

**A. One giant model (End-to-End):**
- *Disadvantage:* Impossible with our current disjointed datasets (masks vs labels vs CSVs).

**B. Separate models (Independent):**
- *Disadvantage:* High inference latency. Error cascading (if detection fails, classification and nutrition fail).

**C. Hybrid Multi-Stage Pipeline (RECOMMENDED):**
- **Stage 1 (Detection/Segmentation):** YOLOv8 or U-Net trained on FoodSeg103 (converted to boxes). Isolates the food items on the plate.
- **Stage 2 (Classification):** ResNet/EfficientNet trained jointly on Food-101 + IndianFood. Takes the cropped outputs of Stage 1 and classifies the specific food name.
- **Stage 3 (Nutrition):** Regression network trained on Nutrition5k. Takes the cropped output, estimates volume/mass, and outputs calories/macros.

================================================

## PART 12 — DATA SPLIT STRATEGY

- **Food-101**: Use existing 75K Train / 25K Test.
- **FoodSeg103**: Use existing 4,983 Train / 2,135 Val.
- **IndianFood**: No existing split. **Create 80/10/10 split (40 Train, 5 Val, 5 Test per class)** using `train_test_split` with a fixed `random_state`.
- **Nutrition5k**: Use existing `rgb_train_ids.txt` and `rgb_test_ids.txt`.

**Crucial**: Never leak test data into train via augmentation.

================================================

## PART 13 — AUGMENTATION STRATEGY

| Augmentation | Classification (Food-101/IndianFood) | Segmentation/Detection | Nutrition5k (Regression) |
|---|---|---|---|
| **Flip (Horizontal)** | SAFE (food is symmetric) | SAFE (Must flip mask/box too) | SAFE |
| **Flip (Vertical)** | SAFE (overhead plates) | SAFE (Must flip mask/box too) | SAFE |
| **Rotation (90/180)** | SAFE | SAFE (Must rotate mask/box too) | SAFE |
| **Color Jitter** | SAFE | SAFE | SAFE |
| **Crop & Resize** | SAFE | SAFE (Adjust boxes) | **RISKY** (Alters perceived volume/mass) |
| **Blur/Noise** | SAFE | SAFE | SAFE |

================================================

## PART 14 — MODEL EVALUATION

- **Detection (Stage 1)**: mAP@0.5 and mAP@0.5:0.95 (Primary).
- **Segmentation (Stage 1)**: Mean Intersection over Union (mIoU), Dice Coefficient.
- **Classification (Stage 2)**: Top-1 Accuracy (Primary), Top-5 Accuracy, F1-Score (for IndianFood imbalance).
- **Nutrition (Stage 3)**: Mean Absolute Error (MAE) for Calories/Grams (Primary).

================================================

## PART 15 — COMPUTE / TRAINING RISKS

- **GPU**: NVIDIA RTX PRO 2000 Blackwell (~16 GB VRAM).
- **Batch Size Risk**: 16 GB VRAM is excellent, but large images (e.g. 512x512) on heavy models (YOLOv8x / ResNet152) will cause OOM (Out of Memory) errors.
- **Recommendation**: 
  - Classification: Batch size 32 or 64. Image size 224x224.
  - Segmentation: Batch size 8 or 16. Image size 512x512.
  - Use PyTorch Mixed Precision (`torch.amp.autocast`) to halve VRAM usage and speed up training.

================================================

## PART 16 — PRE-TRAINING CHECKLIST

### A. MUST FIX BEFORE TRAINING
- [ ] Write a script to convert FoodSeg103 PNG masks to YOLO `.txt` bounding boxes, otherwise Stage 1 object detection cannot be trained.
- [ ] Write a script to generate a strict, reproducible Train/Val/Test split for the IndianFood dataset.
- [ ] Verify that Nutrition5k image IDs actually map to the CSV correctly by writing a PyTorch Dataset class test.

### B. SHOULD FIX BEFORE TRAINING
- [ ] Determine how to merge the 101 classes of Food-101 with the 80 classes of IndianFood into a unified label map (handling duplicates like `samosa`).

### C. NICE TO HAVE
- [ ] Generate augmented versions of the IndianFood dataset to artificially increase the 50 images/class limit to prevent overfitting.

================================================

## PART 17 — FINAL RECOMMENDATION

1. **Top 10 risks**: Missing bounding boxes, Overfitting IndianFood (50 imgs/class), Nutrition5k domain shift (fixed camera), Class name collisions across datasets, Data leakage during splitting, OOM errors during segmentation training, Incompatible annotation formats, Nutrition5k missing images, Pixel-level imbalance in FoodSeg103, Lack of depth camera for end-users.
2. **Highest-priority fixes**: Convert FoodSeg103 masks to YOLO bounding boxes. Split IndianFood properly.
3. **Datasets that should NOT be combined**: FoodSeg103 and IndianFood (Segmentation vs Classification). Nutrition5k and Food-101 (Regression vs Classification).
4. **Datasets that can safely support each other**: Food-101 and IndianFood (can be concatenated for Stage 2 Classification).
5. **Recommended ML architecture**: 3-Stage Pipeline (Detection -> Classification -> Regression).
6. **Recommended dataset-to-model mapping**: 
   - FoodSeg103 -> YOLOv8/U-Net
   - Food-101 + IndianFood -> ResNet/EfficientNet
   - Nutrition5k -> Custom CNN Regression
7. **Recommended train/validation/test strategy**: Respect original splits for FoodSeg103, Food-101, Nutrition5k. Generate fixed stratified split for IndianFood.
8. **Recommended class-balancing strategy**: High augmentation and weight decay for IndianFood. Class weights for FoodSeg103.
9. **Recommended augmentation strategy**: Heavy geometric (flip/rotate) and photometric (color/brightness) across the board. Avoid cropping Nutrition5k to preserve volume cues.
10. **Recommended training order**: 1. Classification (easiest to validate). 2. Segmentation/Detection. 3. Nutrition Regression.
11. **Expected bottlenecks**: Converting masks to bounding boxes. Matching IndianFood classes to Food-101 classes.
12. **Whether additional datasets are actually needed**: No. The addition of Food-101 completely resolves the image volume issue.
13. **Whether 50k+ images is useful**: Yes, 101k images for classification will yield a highly robust, production-ready classifier.
14. **What should be done before the first training run**: Execute the MUST FIX checklist (bounding box generation and IndianFood splitting).
