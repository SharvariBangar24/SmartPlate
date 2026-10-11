# Smart Plate - Experiment Log

## Template
Use the following template for each new experiment:

### Experiment Number: 
- **Objective:** 
- **Dataset:** 
- **Number of images:** 
- **Number of classes:** 
- **Train/validation/test split:** 
- **Model architecture:** 
- **Pretrained weights:** 
- **Epochs:** 
- **Image size:** 
- **Batch size:** 
- **Patience:** 
- **Hardware:** 
- **Training duration:** 
- **Precision:** 
- **Recall:** 
- **mAP@50:** 
- **mAP@50-95:** 
- **Outcome:** 

---

### Experiment 0 - Project Setup
- **Date:** 2026-10-07
- **Dataset:** Not available yet
- **Number of images:** Not available yet
- **Number of classes:** Not available yet
- **Train/validation/test split:** Not available yet
- **Model:** Not available yet
- **Model size:** Not available yet
- **Image size:** Not available yet
- **Epochs:** Not available yet
- **Batch size:** Not available yet
- **Learning rate:** Not available yet
- **Augmentation:** Not available yet
- **Training platform:** Not available yet
- **Training time:** Not available yet
- **Precision:** Not available yet
- **Recall:** Not available yet
- **mAP@50:** Not available yet
- **mAP@50-95:** Not available yet
- **Best epoch:** Not available yet
- **Model weights:** Not available yet
- **Observations:** Initial ML folder structure and templates created. No actual models or data yet.
- **Problems:** Not available yet
- **Changes from previous experiment:** N/A (Initial Setup)

---

### Experiment 1 - Baseline Dataset Preparation
- **Date:** 2026-10-08
- **Dataset:** `smartplate_v1` (IndianFoodNet + SOHL Multi-Dish)
- **Number of images:** 2,742 unique base images
- **Number of classes:** 14 mapped Smart Plate classes
- **Train/validation/test split:** 1,919 / 548 / 275 (70/20/10)
- **Model:** N/A
- **Model size:** N/A
- **Image size:** N/A
- **Epochs:** N/A
- **Batch size:** N/A
- **Learning rate:** N/A
- **Augmentation:** None applied yet. Augmented copies from raw sources strictly discarded.
- **Training platform:** N/A
- **Training time:** N/A
- **Precision:** N/A
- **Recall:** N/A
- **mAP@50:** N/A
- **mAP@50-95:** N/A
- **Best epoch:** N/A
- **Model weights:** N/A
- **Observations:** Successfully processed and mapped 13,413 raw source images down to 2,742 clean, unique images containing 3,835 target bounding boxes. Data leakage is mathematically impossible since identical augmentations were discarded prior to splitting.
- **Problems:** Some target classes (Poori, Vada, Curd) have 0 images. 
- **Changes from previous experiment:** Shifted from dataset raw collection to actual processing and Train/Val/Test preparation. Dataset is READY FOR YOLO TRAINING.

---

### Experiment 1
- **Objective:** Initial YOLO training on the combined dataset
- **Dataset:** Combined YOLO dataset
- **Number of images:** 6,361
- **Number of classes:** 14 (Poori, Vada, and Curd have 0 examples)
- **Train/validation/test split:** 4,452 / 1,272 / 637
- **Model architecture:** YOLO
- **Pretrained weights:** Unknown
- **Epochs:** Unknown
- **Image size:** Unknown
- **Batch size:** Unknown
- **Patience:** Unknown
- **Hardware:** Unknown
- **Training duration:** Unknown
- **Precision:** 0.774
- **Recall:** 0.740
- **mAP@50:** 0.834
- **mAP@50-95:** 0.650
- **Outcome:** Model trained. **Note**: Validation set differed from Experiment 2, so comparisons are not strictly apples-to-apples.

---

### Experiment 2
- **Objective:** YOLO training (second iteration)
- **Dataset:** Combined YOLO dataset
- **Number of images:** 6,361
- **Number of classes:** 14 (Poori, Vada, and Curd have 0 examples)
- **Train/validation/test split:** 4,452 / 1,272 / 637
- **Model architecture:** YOLO
- **Pretrained weights:** Unknown
- **Epochs:** Unknown
- **Image size:** Unknown
- **Batch size:** Unknown
- **Patience:** Unknown
- **Hardware:** Unknown
- **Training duration:** Unknown
- **Precision:** 0.833
- **Recall:** 0.754
- **mAP@50:** 0.793
- **mAP@50-95:** 0.643
- **Outcome:** Precision and Recall improved over Exp 1, but mAPs declined. **Note**: Validation set differed from Experiment 1, so comparisons are not strictly apples-to-apples. Weights were not found in the Drive backup (NOT SAVED).

---

### Experiment 3
- **Objective:** YOLO training with 100 epochs
- **Dataset:** Combined YOLO dataset
- **Number of images:** 6,361
- **Number of classes:** 14 (Poori, Vada, and Curd have 0 examples)
- **Train/validation/test split:** 4,452 / 1,272 / 637
- **Model architecture:** YOLO
- **Pretrained weights:** MISSING (Artifact inaccessible)
- **Epochs:** 100 configured, early stopping at 94 (Best epoch 74)
- **Image size:** MISSING
- **Batch size:** MISSING
- **Patience:** MISSING
- **Hardware:** Colab/Remote (Assumed based on path `/content/drive/...`)
- **Training duration:** 2.238 hours
- **Precision:** MISSING (Cannot verify outputs from Google Drive)
- **Recall:** MISSING (Cannot verify outputs from Google Drive)
- **mAP@50:** MISSING (Cannot verify outputs from Google Drive)
- **mAP@50-95:** MISSING (Cannot verify outputs from Google Drive)
- **Outcome:** Training completed with early stopping at epoch 94. Best checkpoint at epoch 74.
- **Problems/Limitations:** The requested output directory (`/content/drive/MyDrive/Smartplate/experiments/experiment_3_100epochs`) could not be accessed locally. All final validation metrics, per-class metrics, plots, and weights (`best.pt`, `last.pt`, `results.csv`) are explicitly MISSING from the local machine.

---

### Final Model Evaluation (Test Set)
- **Objective:** Evaluate the trained YOLO11s model on the dedicated Test Set.
- **Model architecture:** YOLO11s
- **Model weights location:** `ml/models/best.pt` (18.29 MiB)
- **Test Set Metrics:**
  - **Precision:** 0.8107
  - **Recall:** 0.7601
  - **mAP@50:** 0.8174
  ### Nutrition Integration (End-to-End MVP)
- **Objective:** Map the 14 trained YOLO object classes to estimated macronutrients.
- **Methodology:** 
  - Since full 3D volumetric estimation and NIR scanning are out of scope, a "Bounding-Box-as-a-Serving" heuristic was implemented.
  - A static dictionary (`NUTRITION_DB`) was added to `backend/app.py`.
  - Macros (Calories, Protein, Carbs, Fat) are estimated per standard serving (e.g., 1 Roti = 40g = 120 kcal) using the USDA FoodData Central and Indian Food Composition Tables (IFCT).
- **Outcome:** The Flask API successfully returns aggregated meal totals and per-item macros alongside bounding box coordinates. Missing image requests return HTTP 400.
- **Status:** Integrated.

### Secondary Food Classification Strategy (Pending Colab Training)
- **Objective:** Train a MobileNetV3-Large image classifier to augment the YOLO bounding box classes.
- **Dataset Strategy (Target > 40,000 images):**
  1. `SohlHealth/enhanced-indian-food-classification` (Hugging Face) - **15,404** images.
  2. `IndianFoodNet30` (Roboflow/Local Backup) - **11,487** images.
  3. `sudarshan24/indian-food-images-top-20` (Kaggle) - **7,000** images.
  4. Smart Plate Base YOLO Dataset - **6,361** images.
  - **Total verified combined count:** **40,252** images.
- **Methodology:** Pre-configured in a runnable Jupyter notebook (`ml/notebooks/smartplate_food_classification.ipynb`) to be executed on a Google Colab GPU to bypass local OS restrictions.
- **Status:** Notebook prepared. Training pending execution by user on Colab.
