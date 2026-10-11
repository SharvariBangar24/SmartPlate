# Smart Plate Dataset Research

## 1. Project Dataset Requirement
Target: approximately 20,000–30,000 final usable images for the vision system.

## 2. Dataset Comparison

| Dataset | Images | Classes | Annotation Type | Multi-item | Indian Food Relevance | YOLO Compatibility | License/Restrictions | Recommendation |
|---------|--------|---------|-----------------|------------|-----------------------|--------------------|----------------------|----------------|
| **UEC FOOD-256** | ~31,395 | 256 | Bounding Boxes | Yes | Low/Medium (Mostly Japanese, some universal) | Yes (after COCO/YOLO conversion) | Non-commercial research | Primary Candidate |
| **FoodInsSeg** | 7,118 | 103 (Ingredients) | Instance Segmentation (Masks) | Yes (Highly complex) | Medium (Ingredient level) | Yes (Masks to YOLO seg) | Non-commercial research | Secondary / Augmentation |
| **Indian Food Object Detection** | 1,211 | 14 | Bounding Boxes | Yes | High | Yes (Native YOLO) | MIT (Repo) / Roboflow TOS | Essential |
| **SOHL Multi-Dish** | 377 | 16 | Bounding Boxes | Yes | High | Yes (Native YOLO) | CC-BY / Open | Essential |

## 3. Detailed Dataset Analysis

### UEC FOOD-256
- **Available Images:** ~31,395 images.
- **Classes:** 256 food categories (largely Japanese/Asian).
- **Annotation Type:** Bounding boxes (provided in `bb_info.txt` files per category).
- **Multi-item:** Yes, the dataset includes a `multiple_food.txt` file listing images with two or more items.
- **Suitability:** Highly suitable for bulk object detection training due to the volume of images and bounding boxes, providing a strong foundational model for food detection (plate detection, generic food shapes).
- **Indian Relevance:** Low specific relevance, but universal categories (rice, eggs, salad, soup) provide valuable background patterns.
- **Storage Requirement:** ~1.5 - 2 GB.
- **Advantages:** Massive size, bounding box annotations, covers many universal food textures.
- **Disadvantages:** Highly biased toward Japanese cuisine; annotations are in a custom format requiring conversion to YOLO.

### FoodInsSeg
- **Available Images:** 7,118 images (4,983 Train, 2,135 Test).
- **Classes:** 103 ingredient-level categories (e.g., tomato, meat, rice).
- **Annotation Type:** Instance segmentation masks (COCO format) totaling 119,048 masks.
- **Multi-item:** Extremely dense multi-item support (up to 186 ingredient masks per image).
- **Suitability:** Excellent for Phase 2/Portion Estimation. Segmentation masks allow pixel-area calculation which is critical for volume and nutrition estimation.
- **Indian Relevance:** Medium. It identifies ingredients rather than completed dishes, which is actually highly beneficial for macronutrient mapping.
- **Storage Requirement:** ~2 - 3 GB.
- **Advantages:** First dataset of its kind for ingredient-level *instance segmentation*. Perfect for volume estimation.
- **Disadvantages:** Focuses on ingredients rather than dishes; requires translating COCO masks to YOLO segmentation format.

### Indian Food Object Detection
- **Available Images:** 1,211 images.
- **Classes:** 14 specific Indian dish classes (Biryani, Rice, Roti, Poori, Idli, Vada, Sambhar, Chutney, Daal, Dry Curry, Wet Curry, Salad, Sweet, Curd).
- **Annotation Type:** Bounding boxes.
- **Multi-item:** Yes, supports multi-item Indian meals (Thali style).
- **Suitability:** Essential for Indian specific classification.
- **Storage Requirement:** ~150 - 200 MB.
- **Advantages:** Native YOLO format; highly specific to the project's target cuisine.
- **Disadvantages:** Hosted on Roboflow (requires API access); relatively small.

### SOHL Multi-Dish Indian Food (Locally Verified)
- **Available Images:** 377 images.
- **Classes:** 16 Indian food categories.
- **Annotation Type:** Bounding boxes (YOLO format).
- **Multi-item:** Yes, verified locally (many files contain multiple bbox lines).
- **Suitability:** Excellent core validation dataset.
- **Storage Requirement:** ~48.5 MB.
- **Advantages:** Fully verified, YOLOv8 ready, multi-item Indian dishes.
- **Disadvantages:** Very small (377 images); highly imbalanced class distribution; no pre-defined train/val/test splits.

## 4. Recommended Dataset Combination

To reach the 20,000–30,000 image target while maintaining scientific validity for multi-item Indian food recognition, we must combine datasets strategically using a **Transfer Learning / Pre-training approach**:

1. **Base Pre-training (Generic Food/Plate Localization):** Use **UEC FOOD-256** (~31,395 images). Even though it is Japanese-focused, it trains the YOLO model to understand "what food looks like" and "how to draw bounding boxes around multiple items on a table."
2. **Specialized Fine-tuning (Indian Food):** Combine **Indian Food Object Detection** (1,211) and **SOHL Multi-Dish** (377) to fine-tune the model specifically for the Smart Plate Indian categories.
3. **Portion/Volume Backbone:** Keep **FoodInsSeg** (7,118 images) as a secondary phase dataset to train a YOLOv8-Segmentation model for ingredient-level masking, which is critical for the nutrition estimation pipeline.

This combined pool gives us an initial raw pool of ~40,000 images. After filtering out single-item images from UEC or cleaning bad annotations, we will comfortably land in the 20,000–30,000 high-quality range.

## 5. Risks
- **Class Mismatch:** UEC FOOD-256 has 256 classes, while our Indian datasets have ~14-16. If we naively merge them, the model will output 256 useless Japanese classes. We must carefully map generic UEC classes (like "rice") to our Indian classes ("rice_dish") and drop/ignore irrelevant UEC classes, or strictly use UEC for pre-training only.
- **Incompatible Annotations:** UEC uses a custom `bb_info.txt` format, and FoodInsSeg uses COCO JSON masks. Both require careful parsing scripts to convert to YOLO format.
- **Severe Class Imbalance:** Verified locally on SOHL (e.g., 294 `dry_vegetable` vs 3 `salad`). This will persist and compound when merging datasets.
- **Licensing Issues:** UEC and FoodInsSeg are for non-commercial research only. This limits Smart Plate to an academic/research scope, not a commercial product.
- **Duplicate Images:** There may be identical images scraped from the web across SOHL and the Indian Food Object Detection dataset. Deduplication (e.g., via image hashing) is required before merging.

## 6. Recommended Next Step
**Map and Consolidate Classes:** Before writing any conversion or download scripts, we must define a single, unified "Smart Plate Class Taxonomy" (e.g., 20 final classes) and map the classes from SOHL, Indian Food Obj Det, and the relevant parts of UEC into this single taxonomy.

---
PRIMARY DATASET: UEC FOOD-256 (for bulk volume/pre-training localization)
SECONDARY DATASET(S): Indian Food Object Detection & SOHL Multi-Dish (for Indian-specific fine-tuning), FoodInsSeg (for segmentation/volume estimation).
DATASET TO KEEP ONLY FOR REFERENCE: None. All serve distinct, necessary roles.
DATASET TO AVOID: Pure classification datasets (e.g., Food-101 without bounding boxes).
EXPECTED RAW IMAGE POOL: ~40,101 images.
EXPECTED FINAL USABLE IMAGE RANGE: 20,000–25,000 (after filtering for multi-item and relevant classes).
MAIN REASON: UEC provides the necessary volume to prevent YOLO from overfitting (satisfying the 20k-30k requirement), while the smaller Indian datasets provide the domain-specific accuracy required for the Smart Plate project. FoodInsSeg provides the critical missing link (segmentation masks) needed for volume/nutrition estimation.
