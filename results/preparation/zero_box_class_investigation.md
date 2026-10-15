# Zero-Box Class Investigation Report

This investigation targets the two classes (`steamed bun` and `porridge`) which yielded exactly **0 bounding boxes** in the `FoodSeg103_YOLO` training split. The goal is to determine if these objects were erroneously filtered during bounding box conversion, or if they genuinely do not exist in the training data.

## 1. Investigation Parameters
- **Data Configuration Read**: `D:\SmartPlate\datasets\FoodSeg103_YOLO\data.yaml`
  - `steamed bun` (YOLO Class 103)
  - `porridge` (YOLO Class 104)
- **Source Location Checked**: `D:\SmartPlate\datasets\FoodSeg103\train\masks`
- **Method**: Direct mathematical inspection of the 4,983 binary `.png` semantic segmentation mask arrays. (Mapping: YOLO class `N` corresponds to pixel value `N+1` in the original semantic mask, so `104` for steamed bun and `105` for porridge).

## 2. Findings

A script parsed all 4,983 training masks to extract connected contours for pixel values 104 and 105.

### Class: `steamed bun`
- **Original Train Images containing this class**: 0
- **Original Mask Objects found**: 0
- **Converted YOLO Boxes**: 0
- **Filtered Objects (due to < 5px size limit)**: 0

### Class: `porridge`
- **Original Train Images containing this class**: 0
- **Original Mask Objects found**: 0
- **Converted YOLO Boxes**: 0
- **Filtered Objects (due to < 5px size limit)**: 0

## 3. Conclusion

**The conversion process did NOT lose these objects.** 

The objects were completely absent from the YOLO dataset because **they do not exist in the original FoodSeg103 training split**. 

The pixels corresponding to `steamed bun` and `porridge` simply never appear in any of the 4,983 training masks. They are exclusively present in the validation split (or are completely absent from the dataset entirely). Because they have 0 training examples, it is mathematically impossible for the YOLO model to learn to detect them.

## 4. Recommended Action
Since no bounding boxes can be generated from thin air, we have two options:
1. **Remove them from `data.yaml`**: Drop the classes from the YAML file entirely to prevent the model from evaluating an impossible metric.
2. **Move Validation Images to Train**: Identify the validation images containing `steamed bun` and `porridge` and move a few of them into the training split to provide at least some minimal representation.

*No dataset modifications, augmentations, or model retrainings have been performed during this investigation.*
