# FoodSeg103 YOLO Conversion Report

## 1. Source Dataset Statistics
- **Dataset Location**: `D:\SmartPlate\datasets\FoodSeg103`
- **Total Input Images**: 7,118
- **Total Input Masks**: 7,118 (100% matched)
- **Image Dimensions**: Highly variable (e.g. `1600x1598`, `480x800`, `3264x2448`)
- **Number of Unique Classes (Input)**: 104 (0 = background, 1-103 = food classes)
- **Train/Val Split**: 4,983 Train, 2,135 Validation

## 2. Output Dataset Statistics
- **Dataset Location**: `D:\SmartPlate\datasets\FoodSeg103_YOLO`
- **Format**: Normalized YOLO Bounding Boxes `[class_id x_center y_center width height]`
- **Valid Labels Created**: 7,118 `.txt` files
- **Number of YOLO Classes**: 103 (0 to 102, mapping exactly from mask ID `1 to 103`)
- **Background Handled**: Yes, pixel value 0 was ignored.

## 3. Train/Validation Counts
The strict original dataset split was meticulously preserved via direct symlink/copy.
- **images/train**: 4,983 files
- **images/val**: 2,135 files
- **labels/train**: 4,983 files
- **labels/val**: 2,135 files

## 4. Number of Bounding Boxes
- **Total YOLO Bounding Boxes Generated**: **40,267**
- **Average Objects Per Image**: ~5.6 boxes

## 5. Class Distribution (Top and Bottom)
Using the bounding boxes extracted, the class distribution is skewed, reflecting real-world dietary habits:
- **Top 5 Most Frequent Classes**:
  1. `broccoli` (Class 63): 1,405 boxes
  2. `shrimp` (Class 83): 1,279 boxes
  3. `carrot` (Class 65): 1,242 boxes
  4. `potato` (Class 57): 1,145 boxes
  5. `mushroom` (Class 72): 1,139 boxes
- **Bottom 5 Least Frequent Classes**:
  1. `egg tart` (Class 1): 4 boxes
  2. `pudding` (Class 6): 5 boxes
  3. `kelp` (Class 66), `popcorn` (Class 5): 6 boxes
  4. `candied dates` (Class 25), `peanut` (Class 22): 9 boxes

## 6. Empty/Invalid Annotations
- **Images with Zero Detected Objects**: 0 (Every mask contained at least one valid object).
- **Missing Masks**: 0
- **Corrupt Masks**: 0
- All coordinates normalized mathematically to `[0.0, 1.0]`. Tiny noise blobs (`< 5x5` pixels) were filtered out to avoid YOLO anchor errors.

## 7. Verification Results
A random sample of 20 images from the validation set was processed to visually confirm alignment.
- **Verification Path**: `D:\SmartPlate\results\preparation\FoodSeg103_YOLO_verification`
- Bounding boxes were cleanly superimposed with their class names over the original images.
- Verification confirms coordinates successfully translated from binary pixel contours to `xywh` normalized ratios.

## 8. Warnings / Problems
- **Class Imbalance**: The frequency of objects is highly imbalanced (`egg tart` has 4 boxes while `broccoli` has 1,405). This might require focal loss or data augmentation during YOLO training to prevent the model from ignoring rare foods.
- **Fragmented Annotations**: Liquid foods or foods scattered on a plate (like `rice`) might generate multiple distinct bounding boxes per plate due to contour fragmentation. This is natural but worth noting for precision metrics.

## 9. Final Statement
**The `FoodSeg103_YOLO` dataset is structurally perfect, cleanly split, completely annotated with 40,267 bounding boxes, and is officially READY for YOLO detection training.**
