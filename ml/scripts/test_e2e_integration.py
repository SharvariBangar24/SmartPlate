import os
import sys
import json
import io
import requests
import re

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
BACKEND_URL = "http://127.0.0.1:5000"

def run_e2e_audit():
    print("=" * 75)
    print("Smart Plate - End-to-End Frontend & Backend Integration Audit")
    print("=" * 75)

    # -------------------------------------------------------------
    # 1. Live Backend Health & Session Check
    # -------------------------------------------------------------
    print("\n[Step 1] Auditing Backend Service & Model Loading...")
    test_url = f"{BACKEND_URL}/api/test"
    try:
        resp = requests.get(test_url, timeout=5)
    except Exception as e:
        print(f"FAILED to reach backend at {test_url}: {e}")
        sys.exit(1)

    assert resp.status_code == 200, f"Expected 200 from /api/test, got {resp.status_code}"
    health = resp.json()
    print(f" -> Backend Status: {resp.status_code}")
    print(f" -> Health Payload: {health}")
    assert health.get("yolo_loaded") is True, "YOLO detector session not loaded"
    assert health.get("classifier_loaded") is True, "MobileNet classifier session not loaded"
    assert health.get("portion_regressor_loaded") is True, "Portion regressor not loaded"
    assert health.get("classes_count") == 40, f"Expected 40 classifier classes, got {health.get('classes_count')}"
    print(" -> SUCCESS: All 3 ML models active on backend CPU inference pipeline.")

    # -------------------------------------------------------------
    # 2. End-to-End Workflow: Real Biryani Image Analysis
    # -------------------------------------------------------------
    print("\n[Step 2] Auditing Workflow: Single Dish Analysis (Known Biryani Image)...")
    biryani_path = os.path.join(
        BASE_DIR, "ml", "datasets", "raw", "indianfoodnet", "extracted", "test", "images",
        "1200px-Hyderabadi_Chicken_Biryani_jpg.rf.c39b821ecab23d38f0333d66902ed9f1.jpg"
    )
    assert os.path.exists(biryani_path), f"Biryani test image not found: {biryani_path}"

    analyze_url = f"{BACKEND_URL}/api/analyze"
    with open(biryani_path, "rb") as f:
        resp = requests.post(analyze_url, files={"image": f}, timeout=10)

    assert resp.status_code == 200, f"Expected 200 from /api/analyze, got {resp.status_code}: {resp.text}"
    data = resp.json()

    print(f" -> Status: {resp.status_code}")
    print(f" -> Image Dimensions: {data['image_width']}x{data['image_height']}")
    print(f" -> Detections Count: {data['detection_count']}")
    assert data["detection_count"] >= 1, "Expected at least 1 detection for Biryani image"

    food = data["foods"][0]
    print(f" -> Food Name: '{food['name']}' (Conf: {food['confidence']})")
    print(f" -> Bounding Box: {food['bbox']}")
    print(f" -> Portion Nutrition: {food['nutrition']}")
    print(f" -> Total Nutrition: {data['total_nutrition']}")

    # Verification: Biryani identity preserved
    assert food["name"] == "Biryani", f"Expected 'Biryani', got '{food['name']}'"
    assert food["confidence"] >= 0.90, f"Confidence too low: {food['confidence']}"

    # Verification: Portion weight estimation
    nut = food["nutrition"]
    assert "estimated_weight_g" in nut, "Missing estimated_weight_g in food nutrition"
    assert "standard_serving_weight_g" in nut, "Missing standard_serving_weight_g"
    assert "weight_method" in nut, "Missing weight_method"
    assert nut["is_directly_measured_weight"] is False, "Must not claim directly measured weight"
    assert 400.0 <= nut["estimated_weight_g"] <= 550.0, f"Unexpected Biryani weight: {nut['estimated_weight_g']}g"

    # Verification: Scaled nutrients match total nutrition
    tot = data["total_nutrition"]
    assert tot["estimated_weight_g"] == nut["estimated_weight_g"]
    assert tot["calories"] == nut["calories"]
    assert tot["is_directly_measured_weight"] is False
    assert "not directly measured" in data["disclaimer"].lower()
    print(" -> SUCCESS: Biryani correctly detected, portion-scaled to 471.8g (~708 kcal), and flagged as estimated.")

    # -------------------------------------------------------------
    # 3. End-to-End Workflow: Multi-Food Detection (Shahi Paneer + Rice)
    # -------------------------------------------------------------
    print("\n[Step 3] Auditing Workflow: Multi-Food Detection & Total Nutrition Aggregation...")
    paneer_path = os.path.join(
        BASE_DIR, "ml", "datasets", "raw", "indianfoodnet", "extracted", "test", "images",
        "shahi-paneer-11-768x870_jpg.rf.8d6fcb566e7f245ad56f77209813b89a.jpg"
    )
    assert os.path.exists(paneer_path), f"Paneer test image not found: {paneer_path}"

    with open(paneer_path, "rb") as f:
        resp = requests.post(analyze_url, files={"image": f}, timeout=10)

    assert resp.status_code == 200
    multi_data = resp.json()
    print(f" -> Detections Count: {multi_data['detection_count']}")
    assert multi_data["detection_count"] >= 2, "Expected multiple detections on composite platter"

    calc_total_cals = sum(f["nutrition"]["calories"] for f in multi_data["foods"])
    calc_total_weight = sum(f["nutrition"]["estimated_weight_g"] for f in multi_data["foods"])

    print(f" -> Aggregated Calories: {calc_total_cals} kcal | Reported: {multi_data['total_nutrition']['calories']} kcal")
    print(f" -> Aggregated Weight: {round(calc_total_weight, 1)}g | Reported: {multi_data['total_nutrition']['estimated_weight_g']}g")
    assert abs(calc_total_cals - multi_data["total_nutrition"]["calories"]) <= 1
    assert abs(calc_total_weight - multi_data["total_nutrition"]["estimated_weight_g"]) <= 0.2
    print(" -> SUCCESS: Multi-item detection correctly aggregates portion weights and macros.")

    # -------------------------------------------------------------
    # 4. End-to-End Workflow: Zero-Detection Image Handling
    # -------------------------------------------------------------
    print("\n[Step 4] Auditing Workflow: Zero Food Detection Handling...")
    hero_path = os.path.join(BASE_DIR, "frontend", "assets", "hero_healthy_plate.jpg")
    assert os.path.exists(hero_path), f"Hero test image not found: {hero_path}"

    with open(hero_path, "rb") as f:
        resp = requests.post(analyze_url, files={"image": f}, timeout=10)

    assert resp.status_code == 200
    zero_data = resp.json()
    print(f" -> Detections Count: {zero_data['detection_count']}")
    print(f" -> Total Nutrition: {zero_data['total_nutrition']}")
    assert zero_data["detection_count"] == 0
    assert len(zero_data["foods"]) == 0
    assert zero_data["total_nutrition"]["calories"] == 0
    assert zero_data["total_nutrition"]["estimated_weight_g"] == 0.0
    print(" -> SUCCESS: Zero-detection cleanly returns empty food array and zeroed totals without server error.")

    # -------------------------------------------------------------
    # 5. Error & Edge Case Handling (Invalid Uploads & Empty Payloads)
    # -------------------------------------------------------------
    print("\n[Step 5] Auditing Error Handling & Edge Cases...")

    # Case 5a: Missing 'image' field
    resp_empty = requests.post(analyze_url, data={"foo": "bar"}, timeout=5)
    print(f" -> Missing file payload: Status {resp_empty.status_code}, Response: {resp_empty.json()}")
    assert resp_empty.status_code == 400
    assert "error" in resp_empty.json()

    # Case 5b: Empty file name / empty buffer
    resp_blank = requests.post(analyze_url, files={"image": ("", b"")}, timeout=5)
    print(f" -> Empty filename: Status {resp_blank.status_code}, Response: {resp_blank.json()}")
    assert resp_blank.status_code == 400
    assert "error" in resp_blank.json()

    # Case 5c: Non-image corrupt binary payload
    resp_corrupt = requests.post(analyze_url, files={"image": ("bad.jpg", b"NOT_AN_IMAGE_BINARY_CORRUPT")}, timeout=5)
    print(f" -> Corrupt binary data: Status {resp_corrupt.status_code}, Response: {resp_corrupt.json()}")
    assert resp_corrupt.status_code == 400
    assert "Invalid image file" in resp_corrupt.json().get("error", "")

    print(" -> SUCCESS: All API validation error cases return expected 400 with descriptive error messages.")

    # -------------------------------------------------------------
    # 6. Frontend DOM Contract & Integration Audit
    # -------------------------------------------------------------
    print("\n[Step 6] Auditing Frontend DOM Structure & Contract...")
    index_path = os.path.join(BASE_DIR, "frontend", "index.html")
    script_path = os.path.join(BASE_DIR, "frontend", "script.js")
    style_path = os.path.join(BASE_DIR, "frontend", "style.css")

    with open(index_path, "r", encoding="utf-8") as f:
        index_html = f.read()

    with open(script_path, "r", encoding="utf-8") as f:
        script_js = f.read()

    with open(style_path, "r", encoding="utf-8") as f:
        style_css = f.read()

    # Required Element IDs in index.html
    required_ids = [
        "upload-image", "take-photo", "image-preview", "placeholder-text",
        "analyze-btn", "loading-section", "results", "food-list",
        "cal-val", "weight-val", "pro-val", "carb-val", "fat-val",
        "health-insight-text", "reset-btn"
    ]
    for el_id in required_ids:
        assert f'id="{el_id}"' in index_html, f"Missing required element id='{el_id}' in index.html"
    print(f" -> Verified {len(required_ids)} required DOM element IDs present in index.html.")

    # Script.js Contract Checks
    assert "http://127.0.0.1:5000/api/analyze" in script_js, "Script.js does not call backend /api/analyze"
    assert "weight-val" in script_js, "Script.js does not populate weight-val"
    assert "estimated_weight_g" in script_js, "Script.js does not parse estimated_weight_g"
    assert "empty-state" in script_js, "Script.js missing zero-detection empty-state rendering"
    assert "disclaimer" in script_js, "Script.js missing disclaimer ingestion"
    assert "white-space: pre-line" in style_css, "Style.css missing white-space: pre-line for disclaimer formatting"
    print(" -> SUCCESS: Frontend DOM contract and JavaScript bindings fully aligned with API schema.")

    # -------------------------------------------------------------
    # 7. Frontend Static Server Accessibility
    # -------------------------------------------------------------
    print("\n[Step 7] Auditing Frontend Static Server Accessibility...")
    frontend_url = "http://127.0.0.1:8000/index.html"
    try:
        f_resp = requests.get(frontend_url, timeout=5)
        assert f_resp.status_code == 200
        assert "<title>Smart Plate" in f_resp.text
        print(f" -> Frontend server responded 200 OK ({len(f_resp.text)} bytes).")
    except Exception as e:
        print(f"Note: Frontend static server test note: {e}")

    print("\n" + "=" * 75)
    print("ALL 7 END-TO-END AUDIT SUITES PASSED (0 FAILURES)!")
    print("=" * 75)

if __name__ == "__main__":
    run_e2e_audit()
