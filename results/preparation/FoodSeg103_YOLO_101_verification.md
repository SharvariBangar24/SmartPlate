# FoodSeg103 YOLO 101 Verification Report

This report verifies the creation of the new dataset with ghost classes removed.

## 1. Class Count Discrepancy Notice
- **Expected Old Class Count**: 103
- **Actual Old Class Count**: 105 (Classes were numbered 0 to 104 in the original `data.yaml`)
- **Removed Classes**: 2 (`steamed bun`, `porridge`)
- **Actual New Class Count**: 103 (Classes are now numbered 0 to 102)

*Note: The user instructions requested a final dataset of exactly 101 classes numbered 0 to 100. However, since the original `FoodSeg103_YOLO` dataset contained 105 mapped classes (0-104), removing exactly two ghost classes mathematically results in 103 remaining classes. No other classes were arbitrarily deleted to reach 101, as all remaining 103 classes contain valid bounding boxes. The dataset directory is named `FoodSeg103_YOLO_101` as requested, but contains 103 valid classes.*

## 2. Dataset Verification
- **Old Class Count**: 105
- **New Class Count**: 103
- **Removed Classes**: `steamed bun`, `porridge`

### Split Statistics
- **Training Images**: 4,983
- **Training Labels**: 4,983
- **Validation Images**: 2,135
- **Validation Labels**: 2,135
- **Total Bounding Boxes**: 40,267 (No boxes were lost, as the removed classes had 0 boxes).

### Integrity Checks
- **Every image has a corresponding label file**: Verified.
- **Every label class ID is valid**: Verified. All IDs in the new labels fall strictly between `0` and `102`.
- **No invalid class IDs exist**: Verified. (0 invalid IDs found during parsing).
- **Original Dataset Unchanged**: Verified. `D:\SmartPlate\datasets\FoodSeg103` and `D:\SmartPlate\datasets\FoodSeg103_YOLO` were strictly read-only and were not modified. All changes were isolated to the new `FoodSeg103_YOLO_101` folder.

## 3. Old -> New Class ID Mapping
The complete mapping of the 105 old IDs to the 103 new IDs has been successfully saved to:
`D:\SmartPlate\results\preparation\FoodSeg103_YOLO_101_class_mapping.csv`
