# Zero-Box Validation Candidates Report

This report documents the attempt to locate `steamed bun` and `porridge` within the original FoodSeg103 **validation** split to identify candidates that could be moved to the training split.

## 1. Search Parameters
- **Source Location Checked**: `D:\SmartPlate\datasets\FoodSeg103\validation\masks`
- **Total Masks Scanned**: 2,135
- **Classes Searched**:
  - `steamed bun` (Mask pixel value: 104)
  - `porridge` (Mask pixel value: 105)

## 2. Validation Mask Findings

A Python script mathematically parsed all 2,135 validation masks to extract connected contours for pixel values 104 and 105.

### Class: `steamed bun`
- **Validation Images containing this class**: 0
- **Validation Mask Objects found**: 0
- **Candidate Images**: NONE

### Class: `porridge`
- **Validation Images containing this class**: 0
- **Validation Mask Objects found**: 0
- **Candidate Images**: NONE

## 3. Conclusion

**These two classes are completely absent from the entire downloaded FoodSeg103 dataset.**

They exist neither in the training split nor in the validation split. They appear to be "ghost classes" — categories that were defined in the official dataset label map, but for which no actual annotated images were provided in the public download, or they were held out in a private test set that we do not have access to.

## 4. Final Recommendations

Since there are absolutely zero images of `steamed bun` and `porridge` in the dataset:
- **Recommended Steamed Bun images to move**: 0
- **Recommended Porridge images to move**: 0

**Safest Next Steps**:
1. **Drop the Classes**: We must edit the `data.yaml` file to officially drop/ignore these two classes, reducing our target detection classes from 103 to 101. This prevents the YOLO metrics from indefinitely returning `0.0 mAP` for impossible categories.
2. **Do Not Synthesize Data Yet**: Since our ultimate pipeline combines YOLO with the `IndianFood` and `Food-101` datasets at the classification stage, we can let YOLO simply detect steamed buns and porridge as broad categories (e.g. `bread` or `rice`), and rely on the robust downstream classifier to make the specific correction. 

*No dataset modifications, file copies, or training runs were performed during this investigation.*
