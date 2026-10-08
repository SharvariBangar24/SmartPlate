# FoodSeg103 YOLOv8m Baseline Training Report

## 1. Training Configuration
- **Model**: YOLOv8m (Medium) pretrained on COCO.
- **Image Size**: 640x640
- **Batch Size**: AutoBatch (Maximized for 16GB VRAM)
- **Epochs**: 50 (Early stopped at epoch 42 with patience=10)
- **Best Epoch**: 32
- **Mixed Precision (AMP)**: Enabled
- **Dataset Configuration**: `D:\SmartPlate\datasets\FoodSeg103_YOLO\data.yaml`

## 2. Dataset Information
- **Train Images**: 4,983
- **Validation Images**: 2,135
- **Classes**: 103 specific food items.
- **Augmentation**: Standard YOLO mosaic and photometric transforms.

## 3. Final Metrics (Validation Set)
- **mAP@50**: 0.343
- **mAP@50-95**: 0.286
- **Precision**: 0.472
- **Recall**: 0.341
- **Inference Speed**: ~5.2ms per image (RTX 2000 Ada GPU)

## 4. Per-Class Observations
The model's performance varies radically depending on class representation, directly exposing the severe class imbalance in the FoodSeg103 dataset.

**Best Performing Classes**:
1. `bacon` (420 boxes) - mAP@50: 0.857
2. `blueberry` (217 boxes) - mAP@50: 0.826
3. `potato` (164 boxes) - mAP@50: 0.818
4. `grapefruit` (260 boxes) - mAP@50: 0.792
5. `pork belly` (658 boxes) - mAP@50: 0.786

**Worst Performing Classes (AP = 0)**:
1. `ginger` (1 box) - mAP@50: 0.000
2. `spinach` (3 boxes) - mAP@50: 0.000
3. `pizza` (3 boxes) - mAP@50: 0.000
4. `rice` (4 boxes) - mAP@50: 0.000
5. `ham` (7 boxes) - mAP@50: 0.000

## 5. Confusion/Imbalance Observations
- **Extreme Long-Tail Distribution**: The model effectively achieves >70% AP on frequently appearing, distinctively shaped items (like bacon, potato, and berries). However, it completely fails to learn classes with fewer than 10 examples.
- **Low Recall**: Overall recall is only 34.1%. The model is heavily biased toward the background or common classes, ignoring rare classes entirely.

## 6. Overfitting Observations
- Training loss steadily decreased while validation loss plateaued around epoch 32, triggering early stopping at epoch 42.
- The model is not catastrophically overfitting, but rather **underfitting the minority classes** while memorizing the majority classes. Standard augmentation was insufficient to teach the model what a "ginger" looks like from only 1 image.

## 7. Recommended Next Step
Do not tune hyperparameters on this dataset yet. The primary issue is **data inadequacy for rare classes**.
1. **Oversampling/Augmentation**: Apply heavy offline augmentation specifically to the images containing the worst-performing 30 classes.
2. **Focal Loss / Class Weights**: Introduce class weighting during the loss calculation so the model is heavily penalized for missing rare classes like `spinach` and `rice`.
3. **Data Collection**: If possible, scrape more images of the 0 AP classes. You cannot train a robust object detector on 3 examples of pizza.
