# Indian Food Dataset Expansion Research

## 1. Requirement
We need additional real annotated food images to improve the Indian-domain detection capability of Smart Plate. The target is a realistic, scientifically valid multi-item food detection dataset.

## 2. Dataset Candidates

| Dataset | Images | Classes | Indian Relevance | Annotation | Multi-item | YOLO Comp. | License | Access | Quality | Recommendation |
|---------|--------|---------|------------------|------------|------------|------------|---------|--------|---------|----------------|
| **IndianFoodNet** | ~5,500 | 30 | High | Bounding Boxes | Yes | Yes | CC BY 4.0 | Roboflow | High | **TOP CANDIDATE** |
| **Kaggle YOLO Indian Food (20 classes)** | ~6,000 - 10,000 | 20 | High | Bounding Boxes | Mixed | Yes | CC0 / GPL | Kaggle | Medium | **TOP CANDIDATE** |
| **Khana Dataset** | ~131,000 | 80 | Very High | Classification / Segmentation | Unconfirmed | Unconfirmed | Academic | khana.omkar.xyz | High | **POTENTIAL (Phase 2)** |
| **DataCluster Indian OD** | 5,000+ | Various | High | Bounding Boxes | Yes | Yes | Commercial | Purchase only | High | **REJECTED** |
| **Indian Food Images (Kaggle)** | ~4,000+ | 80 | High | Classification ONLY | No | No | CC0 | Kaggle | Low for OD | **REJECTED** |

## 3. Top Candidates

### 1. IndianFoodNet (Roboflow Universe)
- **Why it's useful:** It is explicitly designed for Indian food object detection. With over 15,000 annotations across 5,500 images, it averages ~3 bounding boxes per image, confirming it strongly supports multi-item detection (Thali style). 
- **Fact vs Estimate:** *Confirmed* image count and YOLO format. 

### 2. Kaggle YOLO Indian Food Datasets
- **Why it's useful:** Several users have compiled YOLO-ready datasets of top 20 Indian foods. While often scraped, they provide bounding boxes for specific dishes (e.g., dosa, idli, biryani).
- **Fact vs Estimate:** *Estimate* on exact image count, as multiple variants exist. Quality requires manual filtering, but it provides a quick raw volume boost.

### 3. Khana Dataset (Academic)
- **Why it's useful:** It is the largest structured Indian food dataset (131k images). 
- **Caveat:** It is primarily built for classification and segmentation, not explicitly bounding-box object detection. If it provides instance segmentation masks, we can programmatically generate bounding boxes from the masks.
- **Fact vs Estimate:** *Confirmed* image count. *Not Verified* if instance-level bounding boxes are easily extractable without running inference.

## 4. Datasets Rejected

- **DataCluster Indian Food Image Object Detection:** Despite having 5,000+ high-quality bounding box images, it is a proprietary commercial dataset requiring a paid license. Not suitable for open academic research without a budget.
- **Indian Food Images (Kaggle - 80 classes):** One of the most famous Kaggle datasets, but it is **classification only**. Using this would require manually drawing bounding boxes or generating pseudo-labels with another model, violating our rule against pseudo-labels.

## 5. Dataset Size Calculation

Calculating a realistic raw and usable pool for purely Indian-domain bounding box datasets:

**Current Confirmed Indian Annotated Images:** ~1,500 (from previous step)
**+ IndianFoodNet:** ~5,500
**+ Kaggle YOLO datasets:** ~6,000 (assuming 1 good variant)
**= RAW TOTAL:** ~13,000 images

**Expected Usable Images:**
- Removing single-item center-focused images: -30%
- Removing duplicates across scraped datasets: -15%
- Filtering out bad annotations: -5%
**REALISTIC USABLE TOTAL:** **~6,500 highly-accurate multi-item Indian food images.**

*(Note: Khana is excluded from this raw calculation until we verify we can extract YOLO bounding boxes from its segmentation data. If we can, Khana adds massive volume).*

## 6. Recommended Dataset Expansion

The strongest combination for Phase 1 Object Detection is:
**Current Smart Plate Data + IndianFoodNet + curated Kaggle YOLO datasets.**
This combination provides purely human-annotated, YOLO-ready bounding boxes specifically for Indian cuisine.

## 7. Important Decision

**Is a final Indian-domain dataset of 20k+ images realistically achievable with publicly available annotated bounding-box datasets?**

**NO.** 
A highly accurate, multi-item, bounding-box specific Indian food dataset of 20k+ images is *not* currently achievable using purely public, free datasets. The publicly available data maxes out around 10,000 - 13,000 raw images, scaling down to ~6,500 usable multi-item images. 

To reach 20k+, we would be forced to:
1. Include single-item classification images and generate pseudo-labels (violates rules).
2. Purchase commercial datasets (DataCluster).
3. Attempt to convert Khana's segmentation masks to bounding boxes (which may be semantic, not instance-level, causing merged boxes).

*We should accept a 5k–10k high-quality Indian dataset paired with a massive UEC generic-food pretraining phase.*

## 8. Next Step

**Define the Final Smart Plate Class Taxonomy:** We must map the classes from our current datasets and IndianFoodNet into a unified taxonomy (e.g., 20 final classes) so we can write merging and filtering scripts.
