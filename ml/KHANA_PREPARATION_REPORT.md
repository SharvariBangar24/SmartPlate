# Khana Dataset Preparation Report

## 1. Extraction and Integrity
- **Source:** `raw/Khana/dataset.tar.gz`
- **Extracted Location:** `C:\Projects\SmartPlate\ml\datasets\processed\khana`
- **Total Valid Images:** 16251
- **Unreadable/Invalid Files:** 0
- **Total Classes Found:** 80
- **Empty Classes:** ['aloo gobi', 'aloo methi', 'aloo mutter', 'aloo paratha', 'amritsari kulcha', 'anda curry', 'balushahi', 'banana chips', 'besan laddu', 'bhindi masala', 'biryani', 'boondi laddu', 'chaas', 'chana masala', 'chapati', 'chicken pizza', 'chicken wings', 'chikki', 'chivda', 'chole bhature', 'dal khichdi', 'dhokla', 'falooda', 'fish curry', 'gajar ka halwa', 'garlic bread', 'garlic naan', 'ghevar', 'grilled sandwich', 'gujhia', 'gulab jamun', 'hara bhara kabab', 'idiyappam', 'idli', 'jalebi', 'kaju katli', 'khakhra', 'kheer', 'kulfi', 'margherita pizza', 'masala papad', 'medu vada', 'modak', 'moong dal halwa', 'murukku', 'mysore pak', 'navratan korma', 'neer dosa', 'onion pakoda', 'palak paneer', 'paneer masala', 'paneer pizza', 'pani puri', 'paniyaram', 'papdi chaat', 'patrode', 'pepperoni pizza', 'phirni', 'poha', 'pongal', 'rajma chawal', 'rava dosa', 'sabudana khichdi', 'sabudana vada', 'samosa', 'seekh kebab', 'set dosa', 'solkadhi', 'steamed momo', 'thali', 'thukpa', 'uttapam']

## 2. Train/Validation/Test Split
A reproducible, class-stratified split was created (80% Train, 10% Val, 10% Test) using fixed random seed (42). 
Instead of duplicating the 6.4 GB of raw images, JSON manifests containing relative paths were generated in `C:\Projects\SmartPlate\ml\datasets\processed\Khana_Splits`.

- **Train:** 12997 images
- **Validation:** 1622 images
- **Test:** 1632 images

## 3. Class Counts
| Class Name | Image Count |
| --- | --- |
| masala dosa | 5894 |
| pav bhaji | 4424 |
| vada pav | 2716 |
| rasgulla | 1151 |
| dabeli | 928 |
| sev puri | 594 |
| misal pav | 543 |
| puri bhaji | 1 |
| aloo gobi | 0 |
| aloo methi | 0 |
| aloo mutter | 0 |
| aloo paratha | 0 |
| amritsari kulcha | 0 |
| anda curry | 0 |
| balushahi | 0 |
| banana chips | 0 |
| besan laddu | 0 |
| bhindi masala | 0 |
| biryani | 0 |
| boondi laddu | 0 |
| chaas | 0 |
| chana masala | 0 |
| chapati | 0 |
| chicken pizza | 0 |
| chicken wings | 0 |
| chikki | 0 |
| chivda | 0 |
| chole bhature | 0 |
| dal khichdi | 0 |
| dhokla | 0 |
| falooda | 0 |
| fish curry | 0 |
| gajar ka halwa | 0 |
| garlic bread | 0 |
| garlic naan | 0 |
| ghevar | 0 |
| grilled sandwich | 0 |
| gujhia | 0 |
| gulab jamun | 0 |
| hara bhara kabab | 0 |
| idiyappam | 0 |
| idli | 0 |
| jalebi | 0 |
| kaju katli | 0 |
| khakhra | 0 |
| kheer | 0 |
| kulfi | 0 |
| margherita pizza | 0 |
| masala papad | 0 |
| medu vada | 0 |
| modak | 0 |
| moong dal halwa | 0 |
| murukku | 0 |
| mysore pak | 0 |
| navratan korma | 0 |
| neer dosa | 0 |
| onion pakoda | 0 |
| palak paneer | 0 |
| paneer masala | 0 |
| paneer pizza | 0 |
| pani puri | 0 |
| paniyaram | 0 |
| papdi chaat | 0 |
| patrode | 0 |
| pepperoni pizza | 0 |
| phirni | 0 |
| poha | 0 |
| pongal | 0 |
| rajma chawal | 0 |
| rava dosa | 0 |
| sabudana khichdi | 0 |
| sabudana vada | 0 |
| samosa | 0 |
| seekh kebab | 0 |
| set dosa | 0 |
| solkadhi | 0 |
| steamed momo | 0 |
| thali | 0 |
| thukpa | 0 |
| uttapam | 0 |

## 4. Khana (80) vs Smart Plate YOLO (14) Class Comparison
The existing Smart Plate YOLO model detects 14 broad categories (e.g., `Curry_Wet`, `Curry_Dry`, `Roti_Naan`, `Sweet`). 
Khana introduces 80 highly specific dishes (e.g., `aloo methi`, `palak paneer`, `jalebi`).

**Mapping Limitations:**
1. **Many-to-One:** Multiple Khana classes (e.g., `aloo gobi`, `chana masala`, `palak paneer`) map to a single YOLO class (`Curry_Wet` or `Curry_Dry`). We cannot train YOLO directly on Khana labels without flattening them, which destroys the fine-grained data.
2. **Missing YOLO Concepts:** YOLO classes like `Snack` or `Sweet` are too broad to assign bounding boxes to from Khana because Khana is whole-image classification, not bounding-box localized. 
3. **No Spatial Data:** Khana lacks bounding boxes entirely.

**Conclusion:** 
Khana should NOT be merged into the YOLO dataset. Instead, Khana will be used to train a secondary Image Classifier. 
When YOLO detects a generic `Curry_Wet` box, that region will be cropped and passed to the Khana-trained classifier to identify if it is specifically `chana masala` or `palak paneer`.
