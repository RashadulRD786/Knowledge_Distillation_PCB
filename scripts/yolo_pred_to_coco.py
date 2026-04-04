import os
import json
import cv2

# -------------------------
# CONFIG
# -------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

PRED_LABEL_DIR = os.path.join(BASE_DIR, "runs/detect/baseline/predict/labels")
IMG_DIR = os.path.join(BASE_DIR, "datasets/HRIPCB/test/images")

MAP_JSON = "image_id_map.json"
OUTPUT_JSON = "predictions_baseline.json"

# -------------------------
# LOAD IMAGE ID MAP (CRITICAL FIX)
# -------------------------
with open(MAP_JSON) as f:
    image_id_map = json.load(f)

print(f"✅ Loaded image_id_map from {MAP_JSON}")

# -------------------------
# GET IMAGE FILES (CONSISTENT ORDER)
# -------------------------
image_files = sorted([
    f for f in os.listdir(IMG_DIR)
    if f.lower().endswith((".jpg", ".jpeg", ".png"))
])

# -------------------------
# CONVERT PREDICTIONS
# -------------------------
coco_preds = []

for img_name in image_files:
    if img_name not in image_id_map:
        continue  # safety check

    img_path = os.path.join(IMG_DIR, img_name)

    # robust label path
    label_name = os.path.splitext(img_name)[0] + ".txt"
    label_path = os.path.join(PRED_LABEL_DIR, label_name)

    if not os.path.exists(label_path):
        continue

    img = cv2.imread(img_path)
    if img is None:
        print(f"⚠️ Skipping unreadable image: {img_name}")
        continue

    h, w = img.shape[:2]
    image_id = image_id_map[img_name]

    with open(label_path, "r") as f:
        for line in f:
            parts = line.strip().split()

            # safety check (some lines may be malformed)
            if len(parts) < 6:
                continue

            cls, x, y, bw, bh, conf = map(float, parts)

            # YOLO → COCO bbox
            x1 = (x - bw / 2) * w
            y1 = (y - bh / 2) * h
            box_w = bw * w
            box_h = bh * h

            coco_preds.append({
                "image_id": image_id,
                "category_id": int(cls),
                "bbox": [x1, y1, box_w, box_h],
                "score": float(conf)
            })

# -------------------------
# SAVE JSON
# -------------------------
with open(OUTPUT_JSON, "w") as f:
    json.dump(coco_preds, f)

print(f"✅ COCO prediction file saved: {OUTPUT_JSON}")
print(f"📊 Total predictions: {len(coco_preds)}")