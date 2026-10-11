# Raw Dataset Validation

## 1. Download Summary
The raw dataset directory structure was created successfully at `C:\Projects\SmartPlate\ml\datasets\`. Due to strict environmental security constraints (e.g., Application Control blocking `pip install roboflow`) and external host requirements (Google Drive authentication, server agreements), automated CLI downloading of external datasets was blocked. SOHL Multi-Dish was successfully preserved and moved to its raw processing directory.

## 2. Dataset-by-Dataset Statistics
- **SOHL Multi-Dish:** 377 images, 377 annotation files.
- **Indian Food Object Detection:** 0 images (ACCESS FAILED)
- **IndianFoodNet:** 0 images (ACCESS FAILED)
- **UEC FOOD-256:** 0 images (ACCESS FAILED)
- **FoodInsSeg:** 0 images (ACCESS FAILED)

## 3. Annotation Validation
**(SOHL Multi-Dish Only):**
- Images without annotations: 0
- Annotations without images: 0
- Malformed annotations: 0
- Invalid bounding boxes (out of bounds): 0
- Empty annotations: 0
- Class IDs outside valid range (0-15): 0

## 4. Duplicate/Corrupt Image Analysis
**(SOHL Multi-Dish Only):**
- Unreadable/corrupt images: 0
- Duplicate filenames: 0
- Image dimensions: All valid, predominantly high-resolution.

## 5. License and Access Notes
- **Roboflow Datasets (IFOD, IndianFoodNet):** Access failed because the environment explicitly blocks the installation of the `roboflow` Python package required to download from Roboflow Universe programmatically.
- **UEC FOOD-256:** Server blocking automated `curl` scripts; requires manual browser download.
- **FoodInsSeg:** Google Drive requires user OAuth/browser session.

## 6. UEC Pretraining Dataset Status
**Status:** UNAVAILABLE (Pending Manual Action). 
It cannot be downloaded automatically. It must remain strictly separated once acquired to serve only as a generic feature-extractor during pretraining.

## 7. FoodInsSeg Segmentation Dataset Status
**Status:** UNAVAILABLE (Pending Manual Action).
It cannot be downloaded automatically. It remains dedicated to Phase 2 (Portion Estimation) and must not be converted for the bounding box task.

## 8. Indian Dataset Status
**Status:** PARTIAL.
Only SOHL Multi-Dish is currently present and fully validated locally. The lack of Indian Food OD and IndianFoodNet severely cripples the raw pool of data required for the 14-class taxonomy.

## 9. Total Raw Image Pool
**Currently Available:** 377 images.
*(Target was ~6,500+ usable images).*

## 10. Problems Found
1. **Critical Data Shortage:** Without manual intervention to bypass the API restrictions on the Roboflow and UEC datasets, the project only has 377 images. This is vastly insufficient to train YOLOv8 on 14 classes without severe overfitting.
2. **Environment Blocks:** The local Windows environment restricts the necessary CLI tools/libraries needed for automated dataset acquisition.

## 11. Recommended Next Step
Because automated downloads are blocked, **the USER must manually download the missing datasets** (Indian Food OD, IndianFoodNet, UEC-256) via a web browser and place the extracted folders into their respective directories inside `C:\Projects\SmartPlate\ml\datasets\raw\`.
Once the data is manually populated, we can proceed to the Annotation Conversion and Merging phase using the approved 14-class taxonomy.
