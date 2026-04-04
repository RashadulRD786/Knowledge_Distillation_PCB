import os
import json
import cv2

# -------------------------
# CONFIG
# -------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_PATH = os.path.join(BASE_DIR, "datasets/HRIPCB")
SPLIT = "test"

images_dir = os.path.join(DATASET_PATH, SPLIT, "images")
labels_dir = os.path.join(DATASET_PATH, SPLIT, "labels")

OUTPUT_JSON = "coco_annotations.json"
MAP_JSON = "image_id_map.json"

# -------------------------
# INIT COCO FORMAT
# -------------------------
coco = {
    "images": [],
    "annotations": [],
    "categories": []
}

# -------------------------
# CLASSES (MUST MATCH TRAINING)
# -------------------------
classes = [
    "missing_hole",
    "mouse_bite",
    "open_circuit",
    "short",
    "spur",
    "spurious_copper"
]

for i, name in enumerate(classes):
    coco["categories"].append({
        "id": i,
        "name": name
    })

# -------------------------
# GET IMAGE LIST (SORTED!)
# -------------------------
image_files = sorted([
    f for f in os.listdir(images_dir)
    if f.lower().endswith((".jpg", ".jpeg", ".png"))
])

# -------------------------
# CREATE CONSISTENT IMAGE MAP
# -------------------------
image_id_map = {name: i for i, name in enumerate(image_files)}

# SAVE mapping (CRITICAL FIX)
with open(MAP_JSON, "w") as f:
    json.dump(image_id_map, f)

print(f"✅ Saved image_id_map → {MAP_JSON}")

# -------------------------
# BUILD COCO
# -------------------------
ann_id = 0

for img_name in image_files:
    img_id = image_id_map[img_name]

    img_path = os.path.join(images_dir, img_name)
    img = cv2.imread(img_path)

    if img is None:
        print(f"⚠️ Skipping unreadable image: {img_name}")
        continue

    h, w = img.shape[:2]

    # Add image info
    coco["images"].append({
        "id": img_id,
        "file_name": img_name,
        "width": w,
        "height": h
    })

    # Label path
    label_path = os.path.join(labels_dir, os.path.splitext(img_name)[0] + ".txt")

    if not os.path.exists(label_path):
        continue

    with open(label_path) as f:
        for line in f:
            cls, x, y, bw, bh = map(float, line.strip().split())

            # YOLO → COCO
            x1 = (x - bw / 2) * w
            y1 = (y - bh / 2) * h
            box_w = bw * w
            box_h = bh * h

            coco["annotations"].append({
                "id": ann_id,
                "image_id": img_id,
                "category_id": int(cls),
                "bbox": [x1, y1, box_w, box_h],
                "area": box_w * box_h,
                "iscrowd": 0
            })

            ann_id += 1

# -------------------------
# SAVE COCO JSON
# -------------------------
with open(OUTPUT_JSON, "w") as f:
    json.dump(coco, f)

print("✅ COCO annotation file created:", OUTPUT_JSON)
print(f"📊 Total images: {len(coco['images'])}")
print(f"📊 Total annotations: {len(coco['annotations'])}")