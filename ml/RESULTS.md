# Smart Plate - ML Results

## 1. Dataset Description
- **Total Images:** 6,361 (Combined YOLO dataset)
- **Classes:** 14 defined classes
- **Splits:** 
  - Training: 4,452
  - Validation: 1,272
  - Testing: 637
- **Note:** Poori, Vada, and Curd currently have zero examples.

## 2. Data Preprocessing
Not available yet

## 3. Training Configuration
Not available yet

## 4. Model Performance
Not available yet

## 5. Precision and Recall Analysis
Not available yet

## 6. mAP Analysis
Not available yet

## 7. Confusion Matrix
Not available yet

## 8. Training and Validation Graphs
Not available yet

## 9. Sample Predictions
Not available yet

## 10. Experiment Comparison
| Metric       | Experiment 1 | Experiment 2 | Experiment 3 |
|--------------|--------------|--------------|--------------|
| Precision    | 0.774        | 0.833        | MISSING      |
| Recall       | 0.740        | 0.754        | MISSING      |
| mAP@50       | 0.834        | 0.793        | MISSING      |
| mAP@50-95    | 0.650        | 0.643        | MISSING      |

*Note: Validation sets differed between Experiments 1 and 2, so this is not a strict apples-to-apples comparison. Experiment 2 weights were not found in the Drive backup and are not saved. Experiment 3 outputs could not be accessed from the provided Google Drive path, so final metrics, charts, and weights are missing.*

## 11. Final Model Selection
The final model chosen for deployment is the YOLO11s model stored at `ml/models/best.pt`.
It was evaluated on the test set and achieved the following metrics:
- **Precision:** 81.07%
- **Recall:** 76.01%
- **mAP@50:** 81.74%
- **mAP@50-95:** 65.47%

## 12. Limitations
- **Nutrition Estimation:** The nutritional analysis uses a "Bounding-Box-as-a-Serving" heuristic. Because volumetric depth sensors and NIR spectroscopy were out of scope, the system assumes each detected bounding box corresponds to one standard serving size (e.g., 1 Roti = 40g, 1 bowl of Dal = 150g). Values are sourced from the USDA and Indian Food Composition Tables (IFCT) but remain static estimates.
- **Classification Granularity:** The 14 YOLO classes are broad (e.g., `Curry_Wet`). Secondary fine-grained classifiers were abandoned due to a lack of sufficiently clean training data within the project deadline.

## 13. Secondary Classification Model (Colab Workflow)
To address the broad YOLO classifications, a secondary MobileNetV3-Large classification model has been prepared for training on Google Colab.
- **Combined Dataset Total:** **40,252 images**
- **Sources:** Enhanced Indian Food Classification (15,404), IndianFoodNet30 (11,487), Kaggle Top-20 (7,000), Smart Plate Base (6,361).
- **Notebook Path:** `ml/notebooks/smartplate_food_classification.ipynb`
- **Status:** Architecture created and datasets verified. Awaiting GPU training run by user.
- **Environment Restrictions:** Local PyTorch execution on Windows may occasionally be blocked by Application Control policies, requiring graceful degradation in the API.

## 13. Future Improvements
Not available yet

## 14. Final Conclusion
Not available yet
