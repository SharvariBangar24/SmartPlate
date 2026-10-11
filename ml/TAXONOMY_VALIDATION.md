# Smart Plate Taxonomy Validation

## 1. Proposed 17 Classes
1. Rice
2. Biryani
3. Roti_Naan
4. Poori
5. Dal
6. Sambar
7. Curry_Wet
8. Curry_Dry
9. Idli
10. Vada
11. Chutney
12. Curd
13. Salad
14. Sweet
15. Snack
16. Egg
17. Fish

## 2. Per-Class Support Analysis

| Class | Source Datasets | Source Classes | Mapping | Est. Instances | Est. Images | Multi-Dataset | Visually Distinct? | Nutrition Mapping? | Risk Level |
|-------|-----------------|----------------|---------|----------------|-------------|---------------|--------------------|--------------------|------------|
| **Rice** | IFOD, SOHL, IFN | `Rice`, `rice_dish` | Merged | 1,500+ | 1,000+ | Yes | Yes (White grains) | Yes | **LOW** |
| **Biryani** | IFOD, IFN | `Biryani` | Direct | ~500 | ~400 | Yes | Yes (Colored/mixed) | Yes | **MEDIUM** |
| **Roti_Naan** | IFOD, SOHL, IFN | `Roti`, `bread_or_Roti_naan` | Merged | 1,500+ | 1,000+ | Yes | Yes (Flatbread) | Yes | **LOW** |
| **Poori** | IFOD | `Poori` | Direct | ~200 | ~150 | No | Yes (Puffed, fried) | Yes | **MEDIUM** |
| **Dal** | IFOD, IFN | `Daal` | Direct | ~400 | ~300 | Yes | No (See Sambar) | Yes | **HIGH** |
| **Sambar** | IFOD, IFN | `Sambhar` | Direct | ~400 | ~300 | Yes | No (See Dal) | Yes | **HIGH** |
| **Curry_Wet** | SOHL, IFOD, IFN | `curry_dish`, `Wet Curry` | Merged | 2,000+ | 1,500+ | Yes | Yes (Liquid/gravy) | No (Needs User Input) | **MEDIUM** |
| **Curry_Dry** | SOHL, IFOD, IFN | `dry_vegetable`, `Dry Curry` | Merged | 2,000+ | 1,500+ | Yes | Yes (Solid/coated) | No (Needs User Input) | **MEDIUM** |
| **Idli** | IFOD, IFN | `Idli` | Direct | ~500 | ~400 | Yes | Yes (White discs) | Yes | **LOW** |
| **Vada** | IFOD, IFN | `Vada` | Direct | ~300 | ~250 | Yes | Yes (Donut shape) | Yes | **LOW** |
| **Chutney** | IFOD, IFN | `Chutney` | Direct | ~800 | ~600 | Yes | Yes (Small bowls) | Yes (Low impact) | **LOW** |
| **Curd** | IFOD | `Curd` | Direct | ~300 | ~250 | No | Yes (White liquid) | Yes | **LOW** |
| **Salad** | SOHL, IFOD | `salad`, `Salad` | Merged | ~150 | ~120 | Yes | Yes (Raw veggies) | Yes | **MEDIUM** |
| **Sweet** | SOHL, IFOD, IFN | `sweet_item`, `Sweet` | Merged | ~400 | ~350 | Yes | No (Too varied) | No (Too varied) | **HIGH** |
| **Snack** | SOHL, Kaggle | `snack_item` | Direct | ~200 | ~150 | Yes | No (Too varied) | No (Too varied) | **HIGH** |
| **Egg** | SOHL, UEC | `eggs` | Direct | ~100 | ~80 | Yes | Yes | Yes | **MEDIUM** |
| **Fish** | SOHL | `fish_dish` | Direct | ~150 | ~120 | No | Yes (Often distinct) | Yes | **MEDIUM** |

*(Note: IFN = IndianFoodNet, IFOD = Indian Food Object Detection)*

## 3. Class Imbalance Analysis
- **Strong Support:** `Curry_Wet`, `Curry_Dry`, `Rice`, `Roti_Naan`. These will dominate the dataset.
- **Moderate Support:** `Biryani`, `Idli`, `Chutney`, `Sweet`, `Dal`, `Sambar`.
- **Weak Support:** `Egg`, `Salad`, `Fish`, `Poori`, `Snack`. These may struggle to achieve high mAP without Focal Loss or heavy augmentation.
- **Classes that may cause serious class imbalance:** `Curry_Wet` and `Curry_Dry`. They naturally appear in almost every Indian meal, dwarfing classes like `Egg`.

## 4. Visual Distinguishability (Special Attention Pairs)
- **Rice vs Biryani:** Visually distinct. Biryani has yellow/orange hues, long grains, and visible embedded spices/meat. Object detection handles this well if the bounding boxes are tight.
- **Roti_Naan vs Poori:** Highly distinct. Poori is smaller, deep golden-brown, and puffed. Roti/Naan is larger, flatter, with char marks.
- **Dal vs Sambar:** **SEVERE RISK.** Both are served in identical small steel bowls and are yellow/brown liquids. Sambar may have visible vegetable chunks, while Dal is smooth, but at low resolution, YOLO will confuse them constantly.
- **Curry_Wet vs Curry_Dry:** Excellent visual separation. Gravy (wet) vs coated/sautéed (dry). 
- **Egg vs Fish:** Visually distinct. Sliced boiled eggs or egg curries look very different from fried fish or fish curries.
- **Sweet vs Snack:** **SEVERE RISK.** A `Sweet` class is a semantic concept, not a visual one. A Gulab Jamun (brown sphere) shares zero visual features with a Jalebi (orange spiral). Similarly, `Snack` groups Samosas (triangles) with Pakoras (irregular blobs). YOLO trains on consistent visual features; forcing it to learn a generic "Snack" class will result in very low precision and random guessing.

## 5. Nutrition Mapping Suitability
- **Straightforward:** Rice, Roti, Idli, Dal, Curd, Egg.
- **Ambiguous:** `Curry_Wet` and `Curry_Dry`. We must rely on the frontend app to ask the user "Is this Veg or Non-Veg?"
- **Impossible:** `Sweet` and `Snack`. A Gulab Jamun has completely different calories than a Barfi. A Samosa is vastly different from roasted peanuts. Grouping them destroys nutritional accuracy.

## 6. Risky Classes
1. **Sweet:** Definition too broad. Visually inconsistent. Nutritionally useless.
2. **Snack:** Definition too broad. Visually inconsistent. Nutritionally useless.
3. **Dal and Sambar:** Visually ambiguous.

## 7. Recommended Changes
To ensure the taxonomy is viable for YOLO and accurate for Smart Plate:
1. **Drop "Sweet" and "Snack" as generic classes.** If IndianFoodNet or Kaggle provides specific annotations for `Gulab_Jamun` or `Samosa`, use those specific classes instead. If there is not enough data for specific sweets/snacks, *exclude them from the first training experiment* rather than polluting the model with a generic guessing class.
2. **Merge Dal and Sambar into `Dal_Sambar`:** Because YOLO will likely fail to distinguish them in small bowls, merge them visually. In the app frontend, ask the user to clarify or simply map the average nutritional value of lentil soups.

## 8. Final Verdict
**TAXONOMY STATUS: APPROVED WITH CHANGES**

The base logic is excellent, but the generic classes will harm the model's overall mAP and provide zero nutritional value. The smallest possible change is dropping the two overly broad semantic classes and merging the two visually identical lentil soups.

## 9. Dataset Construction Requirements
- Filter out all images where bounding boxes are assigned to `Snack` or `Sweet` unless they can be re-mapped to a specific item.
- Combine `Dal` and `Sambar` annotations into `Dal_Sambar`.

---
FINAL APPROVED CLASS LIST:
1. Rice
2. Biryani
3. Roti_Naan
4. Poori
5. Dal_Sambar
6. Curry_Wet
7. Curry_Dry
8. Idli
9. Vada
10. Chutney
11. Curd
12. Salad
13. Egg
14. Fish
*(Reduced to 14 highly reliable, visually distinct, and nutritionally mappable classes).*
