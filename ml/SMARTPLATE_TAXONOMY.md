# Smart Plate Final Class Taxonomy

## 1. Design Principles
The Smart Plate taxonomy must bridge the gap between computer vision capabilities and nutritional accuracy. We adhere to these principles:
- **Visual Distinguishability:** Classes must be visually distinct in a 2D bounding box (e.g., distinguishing Chicken Curry from Paneer Curry visually is near impossible, hence a unified `Curry_Wet` class).
- **Nutritional Variance:** Foods with vastly different calorie/macro densities must not be merged if annotations support separating them (e.g., separating deep-fried `Poori` from roasted `Roti_Naan`, separating `Rice` from `Biryani`).
- **Data Support:** Classes with fewer than 10 examples across all datasets are dropped or merged to prevent YOLO from failing to converge on them.

## 2. Source Class Analysis

| Smart Plate Class | SOHL Source | Indian Food OD Source | Other Sources | Mapping Confidence | Nutrition Usefulness |
|-------------------|-------------|-----------------------|---------------|--------------------|----------------------|
| **Rice** | `rice_dish` | `Rice` | IndianFoodNet | High | High (Base carb) |
| **Biryani** | NOT MAPPED* | `Biryani` | IndianFoodNet | High | High (High fat/cal) |
| **Roti_Naan** | `bread_or_Roti_naan`| `Roti` | IndianFoodNet | High | High (Base carb) |
| **Poori** | NOT MAPPED | `Poori` | - | High | High (Deep fried) |
| **Dal** | NOT MAPPED* | `Daal` | IndianFoodNet | High | High (Protein) |
| **Sambar** | NOT MAPPED* | `Sambhar` | IndianFoodNet | High | High (Veg/Lentil) |
| **Curry_Wet** | `curry_dish`| `Wet Curry` | IndianFoodNet | High | Medium (Needs clarification) |
| **Curry_Dry** | `dry_vegetable`| `Dry Curry` | IndianFoodNet | High | Medium (Needs clarification) |
| **Idli** | NOT MAPPED* | `Idli` | IndianFoodNet | High | High (Fermented carb) |
| **Vada** | NOT MAPPED* | `Vada` | IndianFoodNet | High | High (Deep fried) |
| **Chutney** | NOT MAPPED* | `Chutney` | IndianFoodNet | Medium | Low (Low volume) |
| **Curd** | NOT MAPPED* | `Curd` | - | Medium | Medium (Probiotic/Fat) |
| **Salad** | `salad` | `Salad` | - | High | High (Fibre) |
| **Sweet** | `sweet_item` | `Sweet` | IndianFoodNet | High | High (Sugar/Fat) |
| **Snack** | `snack_item` | NOT MAPPED | Samosa (Kaggle) | Medium | High (Fried calories) |
| **Egg** | `eggs` | NOT MAPPED | UEC-256 (pretrain)| High | High (Protein) |
| **Fish** | `fish_dish` | NOT MAPPED | - | High | High (Protein/Fat) |

*\*SOHL contains ambiguous classes (`south_indian_breakfast`, `Dal_or_sambar`) that conflict with the more precise Indian Food OD classes. These ambiguous SOHL classes are excluded rather than forcefully mapped.*

## 3. Option A — Conservative
**11 Classes:** Rice, Roti_Naan, Dal, Sambar, Curry_Wet, Curry_Dry, Idli, Vada, Chutney, Salad, Sweet.
- **Advantages:** Highly reliable annotations. Every class has hundreds/thousands of examples. Minimal class imbalance.
- **Disadvantages:** Drops Biryani and Poori, forcing them to be mapped as Rice and Roti. This causes a massive nutritional estimation error (underestimating calories).
- **Nutrition/Detection Difficulty:** Very easy to detect, but poor nutritional resolution.

## 4. Option B — Balanced
**17 Classes:** Rice, Biryani, Roti_Naan, Poori, Dal, Sambar, Curry_Wet, Curry_Dry, Idli, Vada, Chutney, Curd, Salad, Sweet, Snack, Egg, Fish.
- **Advantages:** Excellent balance. Keeps critical nutritional distinctions (Biryani vs Rice, Poori vs Roti) alive because IFOD and IndianFoodNet support them natively. Drops the messy, ambiguous classes from SOHL.
- **Disadvantages:** SOHL only provides 12 examples of Eggs and 3 of Salad, so these classes will rely heavily on Kaggle/IndianFoodNet or generic UEC pretraining to survive class imbalance.
- **Nutrition/Detection Difficulty:** Nutrition mapping is highly accurate for starches and sweets. Detection difficulty is moderate for distinguishing wet vs dry curries.

## 5. Option C — Detailed
**25+ Classes:** Rice, Biryani, Roti, Naan, Poori, Paratha, Dal, Sambar, Paneer Curry, Chicken Curry, Mutton Curry, Veg Dry, Idli, Vada, Dosa, Uttapam, Chutney, Curd, Salad, Gulab Jamun, Jalebi, Samosa, Egg, Fish, Pickle.
- **Advantages:** Perfect nutritional mapping if detected correctly.
- **Disadvantages:** Computer vision cannot reliably distinguish Chicken Curry from Mutton Curry or Paneer Curry from tofu based purely on a top-down bounding box. Extreme data imbalance for specific sweets (e.g., Jalebi).
- **Nutrition/Detection Difficulty:** Impossible to detect reliably with current datasets.

## 6. Recommended Taxonomy
**OPTION B (Balanced - 17 Classes)** is strictly recommended. 
It preserves vital nutritional distinctions (frying vs roasting, complex mixed rice vs plain rice) while ensuring the YOLO model doesn't get confused by visually identical dishes (like chicken vs mutton curry). 

## 7. Complete Class Mapping
The complete, explicit mapping of source dataset classes to the final Smart Plate classes has been saved to:
`C:\Projects\SmartPlate\ml\SMARTPLATE_CLASS_MAPPING.csv`

## 8. Excluded Classes
The following source classes are explicitly excluded from the final training set:
- **SOHL `south_indian_breakfast`**: Too generic. Conflicts with the precise `Idli` and `Vada` classes from IFOD.
- **SOHL `Dal_or_sambar`**: Only 3 instances exist in SOHL. Using it breaks the precise `Dal` vs `Sambar` split in IFOD.
- **SOHL `accompaniment`**: Ambiguous. Could mean pickle, chutney, or curd.
- **SOHL `drink`, `pasta`, `soup`, `fruits`**: Not part of the core Indian Thali plate detection target; `fruits` is too broad for specific calorie mapping.

## 9. Class Imbalance Risks
- **Curry_Wet and Curry_Dry** will dominate the dataset (highest frequency).
- **Salad, Egg, and Fish** have severe under-representation in the currently downloaded SOHL dataset. We will heavily rely on Kaggle datasets and IndianFoodNet to balance these classes, or utilize Focal Loss during YOLO training.

## 10. Nutrition Mapping Considerations
Because we mapped visually identical curries into `Curry_Wet` and `Curry_Dry`, the Smart Plate system will not know if a `Curry_Wet` is Chicken (high protein) or Veg (high carb/fat). 
**Solution:** The YOLO model will output `Curry_Wet`, and the frontend App must prompt the user with a quick clarification popup: *"We detected 1 Wet Curry. Is this Veg or Non-Veg?"* This hybrid approach ensures 100% nutritional accuracy without demanding impossible tasks from the vision model.

## 11. Final Recommendation
Adopt **Option B** (17 classes). It provides the highest nutritional value without sacrificing Mean Average Precision (mAP) during YOLO training.

## 12. Next Step
The final action required before moving to model training architecture is to write a Python script that parses the raw annotations from the candidate datasets (using the mapping CSV) to filter and build the final, unified Smart Plate dataset.
