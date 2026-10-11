import os
import sys
import json
import io
import cv2
import numpy as np

# Add backend directory to sys.path
sys.path.insert(0, os.path.abspath('backend'))

from app import (
    app, yolo_session, classifier_session, portion_model, NUTRITION_DB,
    classify_crop, CATEGORY_REFINEMENT_MAP, CLASSIFIER_CLASSES,
    estimate_food_portion, parse_standard_serving_weight
)

def run_tests():
    print("=" * 70)
    print("Smart Plate Backend ONNX Inference & API Verification Test Suite")
    print("=" * 70)

    # 1. Model Loading Validation
    print("\n[Test 1] Validating ONNX sessions and portion model...")
    assert yolo_session is not None, "YOLO session is None!"
    assert classifier_session is not None, "Classifier session is None!"
    assert portion_model is not None, "Portion estimation model is None!"
    print(f" -> YOLO session loaded successfully: {yolo_session.get_inputs()[0].name} -> {yolo_session.get_outputs()[0].name}")
    print(f" -> MobileNet session loaded successfully: {classifier_session.get_inputs()[0].name} -> {classifier_session.get_outputs()[0].name}")
    print(f" -> Portion model loaded successfully: {type(portion_model).__name__} (features: {getattr(portion_model, 'feature_names_in_', 'N/A')})")

    client = app.test_client()

    # 2. Test GET /api/test
    print("\n[Test 2] Testing GET /api/test endpoint...")
    resp = client.get('/api/test')
    assert resp.status_code == 200, f"Expected 200, got {resp.status_code}"
    test_data = resp.get_json()
    print(f" -> Status: {resp.status_code}, Response: {test_data}")
    assert test_data["yolo_loaded"] is True
    assert test_data["classifier_loaded"] is True
    assert test_data["portion_regressor_loaded"] is True
    assert test_data["classes_count"] == 40

    # 3. Test POST /api/analyze without file (Error handling 400)
    print("\n[Test 3] Testing POST /api/analyze without image (400 check)...")
    resp = client.post('/api/analyze')
    assert resp.status_code == 400, f"Expected 400, got {resp.status_code}"
    print(f" -> Status: {resp.status_code}, Response: {resp.get_json()}")

    # 4. Test POST /api/analyze with Real Food Image (Biryani)
    print("\n[Test 4] Testing POST /api/analyze on Real Biryani Image...")
    sample_img_path = r'ml\datasets\raw\indianfoodnet\extracted\test\images\1200px-Hyderabadi_Chicken_Biryani_jpg.rf.c39b821ecab23d38f0333d66902ed9f1.jpg'
    assert os.path.exists(sample_img_path), f"Sample image not found: {sample_img_path}"

    with open(sample_img_path, 'rb') as f:
        img_bytes = f.read()

    data = {
        'image': (io.BytesIO(img_bytes), 'test_biryani.jpg')
    }
    resp = client.post('/api/analyze', data=data, content_type='multipart/form-data')
    assert resp.status_code == 200, f"Expected 200, got {resp.status_code}: {resp.data}"
    result = resp.get_json()

    print(f" -> Status: {resp.status_code}")
    print(f" -> Image Dimensions: {result['image_width']}x{result['image_height']}")
    print(f" -> Detected Foods Count: {result['detection_count']}")
    for i, food in enumerate(result['foods']):
        print(f"    Food #{i+1}: {food['name']} (Conf: {food['confidence']}) | BBox: {food['bbox']}")
        print(f"       Nutrition: {food['nutrition']}")
    print(f" -> Total Nutrition: {result['total_nutrition']}")

    # Critical Assertions: Biryani must NOT be misclassified as fried_rice or white_rice
    assert result["detection_count"] >= 1, "Expected at least 1 detection on Biryani image"
    biryani_food = result["foods"][0]
    assert biryani_food["name"] == "Biryani", f"CRITICAL FAILURE: Expected 'Biryani', got '{biryani_food['name']}'"
    assert biryani_food["name"] != "fried_rice", "CRITICAL FAILURE: Biryani was misclassified as fried_rice!"
    assert biryani_food["name"] != "white_rice", "CRITICAL FAILURE: Biryani was misclassified as white_rice!"
    assert biryani_food["confidence"] >= 0.90, f"Expected high confidence for Biryani, got {biryani_food['confidence']}"
    print(" -> SUCCESS: Biryani correctly detected and preserved as 'Biryani' (never overridden by fried_rice)!")

    # 5. Focused Unit Test: Biryani preservation in classify_crop even when MobileNet predicts fried_rice
    print("\n[Test 5] Focused Unit Test: classify_crop strictly preserves 'Biryani'...")
    b_img = cv2.imread(sample_img_path)
    b_crop = b_img[1:636, 0:632]
    # On this exact crop, MobileNet raw output is fried_rice with 0.999 probability:
    pred_name, pred_conf = classify_crop(b_crop, "Biryani", 0.967)
    assert pred_name == "Biryani", f"Expected 'Biryani', got '{pred_name}'"
    assert pred_conf == 0.967, f"Expected confidence 0.967, got {pred_conf}"
    print(f" -> Verified: classify_crop returned name='{pred_name}', conf={pred_conf} (no override allowed)")

    # 6. Focused Unit Test: Valid coarse-category refinement
    print("\n[Test 6] Focused Unit Test: Valid coarse-category refinement...")
    import glob
    paneer_matches = glob.glob(r'ml\datasets\raw\indianfoodnet\extracted\test\images\*hahi*paneer*.jpg')
    assert len(paneer_matches) > 0, "No Shahi Paneer test image found"
    paneer_path = paneer_matches[0]
    paneer_img = cv2.imread(paneer_path)
    # Pass coarse YOLO category 'Curry_Wet'
    refined_name, refined_conf = classify_crop(paneer_img, "Curry_Wet", 0.70)
    print(f" -> Testing coarse 'Curry_Wet' on ShahiPaneer crop ({os.path.basename(paneer_path)[:35]}):")
    print(f"    Refined Result: '{refined_name}' (conf: {round(refined_conf, 3)})")
    assert refined_name in CATEGORY_REFINEMENT_MAP["curry_wet"], f"Expected refined class in {CATEGORY_REFINEMENT_MAP['curry_wet']}, got '{refined_name}'"
    assert refined_name == "kadai_paneer", f"Expected 'kadai_paneer', got '{refined_name}'"
    assert refined_conf >= 0.90, f"Expected high confidence, got {refined_conf}"
    print(" -> SUCCESS: Valid coarse category 'Curry_Wet' successfully refined to compatible fine-grained 'kadai_paneer'!")

    # 7. Focused Unit Test: Incompatible fine-category prediction rejected
    print("\n[Test 7] Focused Unit Test: Incompatible fine-category prediction rejected...")
    # On Aloo-Gobi crop, MobileNet predicts 'pakode' with 0.884 confidence.
    ag_matches = glob.glob(r'ml\datasets\raw\indianfoodnet\extracted\test\images\*Aloo*Gobi*.jpg')
    assert len(ag_matches) > 0, "No Aloo Gobi test image found"
    ag_path = ag_matches[0]
    ag_img = cv2.imread(ag_path)

    # Test 7a: YOLO category is 'Curry_Dry'. MobileNet predicts 'pakode' (which is NOT in Curry_Dry).
    rej_name_1, rej_conf_1 = classify_crop(ag_img, "Curry_Dry", 0.75)
    print(f" -> Pair 1: Crop predicts 'pakode', YOLO class is 'Curry_Dry':")
    print(f"    Result: '{rej_name_1}' (conf: {rej_conf_1})")
    assert rej_name_1 == "Curry_Dry", f"Incompatible fine class leaked through! Expected 'Curry_Dry', got '{rej_name_1}'"
    assert rej_conf_1 == 0.75

    # Test 7b: YOLO category is 'Roti_Naan'. MobileNet predicts 'pakode' (which is NOT in Roti_Naan).
    rej_name_2, rej_conf_2 = classify_crop(ag_img, "Roti_Naan", 0.85)
    print(f" -> Pair 2: Crop predicts 'pakode', YOLO class is 'Roti_Naan':")
    print(f"    Result: '{rej_name_2}' (conf: {rej_conf_2})")
    assert rej_name_2 == "Roti_Naan", f"Incompatible fine class leaked through! Expected 'Roti_Naan', got '{rej_name_2}'"
    assert rej_conf_2 == 0.85

    print(" -> SUCCESS: Incompatible fine-grained predictions correctly rejected; original YOLO classes strictly preserved!")

    # 8. Focused Unit Test: Unconditional override rule eliminated
    print("\n[Test 8] Focused Unit Test: Unconditional high-confidence override eliminated...")
    # Feeding the Shahi Paneer crop (MobileNet predicts kadai_paneer with 0.999) under YOLO class 'Salad'
    # Under old logic, Rule 2 would overwrite 'Salad' with 'kadai_paneer' because conf >= 0.70.
    res_name, res_conf = classify_crop(paneer_img, "Salad", 0.90)
    print(f" -> Testing high-conf crop (0.999) under non-refinable category 'Salad':")
    print(f"    Result: '{res_name}' (conf: {res_conf})")
    assert res_name == "Salad", f"Unconditional override fired! Expected 'Salad', got '{res_name}'"
    assert res_conf == 0.90
    print(" -> SUCCESS: Unconditional override rule eliminated. Categories outside refinement map are never altered.")

    # 9. API Schema & Frontend Contract Validation
    print("\n[Test 9] Validating API JSON Schema conformance with frontend/script.js...")
    required_top_keys = ["image_width", "image_height", "detection_count", "foods", "total_nutrition", "disclaimer"]
    for k in required_top_keys:
        assert k in result, f"Missing top-level key '{k}' in API response"

    required_nutri_keys = ["calories", "protein", "carbs", "fat"]
    for k in required_nutri_keys:
        assert k in result["total_nutrition"], f"Missing nutrient key '{k}' in total_nutrition"
        assert isinstance(result["total_nutrition"][k], (int, float)), f"Nutrient '{k}' must be numeric"

    # Verify portion weight fields in total_nutrition
    assert "estimated_weight_g" in result["total_nutrition"]
    assert isinstance(result["total_nutrition"]["estimated_weight_g"], (int, float))
    assert result["total_nutrition"]["is_directly_measured_weight"] is False
    assert "not directly measured" in result["disclaimer"].lower()

    for food in result["foods"]:
        assert "name" in food and isinstance(food["name"], str)
        assert "confidence" in food and isinstance(food["confidence"], float)
        assert "bbox" in food and len(food["bbox"]) == 4
        assert "nutrition" in food
        for k in required_nutri_keys:
            assert k in food["nutrition"], f"Missing food nutrient key '{k}'"
            assert isinstance(food["nutrition"][k], (int, float))
        # Check portion fields in each food
        assert "estimated_weight_g" in food["nutrition"]
        assert "standard_serving_weight_g" in food["nutrition"]
        assert "weight_method" in food["nutrition"]
        assert food["nutrition"]["is_directly_measured_weight"] is False

    print(" -> SUCCESS: Full API JSON contract conforms to frontend specifications with portion scaling!")

    # 10. Focused Unit Test: Portion Estimation & Scaled Nutrition
    print("\n[Test 10] Focused Unit Test: Portion estimation & scaled nutrition calculation...")
    # Test Biryani on 640x640:
    biryani_std_g = parse_standard_serving_weight(NUTRITION_DB["biryani"])
    b_weight, b_method, b_measured = estimate_food_portion([0, 0, 600, 600], 640, 640, "Biryani", biryani_std_g)
    print(f" -> Biryani (600x600 on 640x640): weight={b_weight}g, method='{b_method}', measured={b_measured}")
    assert 400.0 <= b_weight <= 600.0, f"Expected Biryani portion ~450-500g, got {b_weight}g"
    assert b_method == "fpb_geometric_regressor"
    assert b_measured is False

    # Check scaling logic
    scale = b_weight / biryani_std_g
    scaled_cal = round(NUTRITION_DB["biryani"]["cal"] * scale)
    assert scaled_cal > NUTRITION_DB["biryani"]["cal"], "Expected scaled calories to be higher for large plate portion"
    print(f" -> Scaled Calories: {scaled_cal} kcal (reference: {NUTRITION_DB['biryani']['cal']} kcal, scale: {round(scale, 2)})")

    # 11. Focused Unit Test: Safeguards for Invalid Bboxes and Degenerate Geometry
    print("\n[Test 11] Focused Unit Test: Safeguards for invalid bboxes and degenerate geometry...")
    # Case a: Inverted coordinates (x2 <= x1)
    w_inv, m_inv, meas_inv = estimate_food_portion([500, 500, 100, 100], 640, 640, "Biryani", biryani_std_g)
    assert w_inv == biryani_std_g, f"Expected fallback to standard serving {biryani_std_g}, got {w_inv}"
    assert "fallback" in m_inv
    assert meas_inv is False
    print(f" -> Inverted coords handled safely: weight={w_inv}g, method='{m_inv}'")

    # Case b: Degenerate width/height (0 area)
    w_zero, m_zero, _ = estimate_food_portion([100, 100, 100, 200], 640, 640, "Biryani", biryani_std_g)
    assert w_zero == biryani_std_g
    print(f" -> Zero-width box handled safely: weight={w_zero}g, method='{m_zero}'")

    # Case c: Invalid image dimensions (0x0)
    w_img0, m_img0, _ = estimate_food_portion([0, 0, 100, 100], 0, 0, "Biryani", biryani_std_g)
    assert w_img0 == biryani_std_g
    print(f" -> Zero image dimensions handled safely: weight={w_img0}g, method='{m_img0}'")

    # Case d: None/malformed bbox
    w_none, m_none, _ = estimate_food_portion(None, 640, 640, "Biryani", biryani_std_g)
    assert w_none == biryani_std_g
    print(f" -> None bbox handled safely: weight={w_none}g, method='{m_none}'")

    # 12. Focused Unit Test: Unmapped classes and extreme predictions
    print("\n[Test 12] Focused Unit Test: Unmapped food class and plausible bounds...")
    # Food class with no FPB equivalent (defaults to NaN class_id)
    w_unmapped, m_unmapped, _ = estimate_food_portion([50, 50, 300, 300], 640, 640, "completely_unmapped_food", 150.0)
    assert 15.0 <= w_unmapped <= 1200.0, f"Predicted weight out of bounds: {w_unmapped}"
    assert m_unmapped == "fpb_geometric_regressor"
    print(f" -> Unmapped food class handled smoothly via NaN branch: weight={w_unmapped}g")

    # Extreme tiny box (1x1 pixel) -> should be clamped to >= 15g
    w_tiny, _, _ = estimate_food_portion([10, 10, 11, 11], 640, 640, "Biryani", biryani_std_g)
    assert w_tiny >= 15.0, f"Expected clamped minimum >= 15g, got {w_tiny}"
    print(f" -> Tiny box clamped to minimum: {w_tiny}g")

    # 13. Focused Unit Test: Direct Measurement Distinction & Disclaimer Validation
    print("\n[Test 13] Validating explicit distinction from directly measured weight...")
    assert biryani_food["nutrition"]["is_directly_measured_weight"] is False
    assert result["total_nutrition"]["is_directly_measured_weight"] is False
    assert "fpb" in result["disclaimer"].lower() or "portion" in result["disclaimer"].lower()
    assert "not directly measured" in result["disclaimer"].lower()
    print(" -> SUCCESS: Clear distinction between estimated and measured weight strictly verified!")

    print("\n" + "=" * 70)
    print("ALL 13 TEST SUITES PASSED SUCCESSFULLY (0 FAILURES)!")
    print("=" * 70)

if __name__ == '__main__':
    run_tests()
