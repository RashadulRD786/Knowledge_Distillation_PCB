"""
select_strict_qualitative.py
============================
Scans the test set and finds the BEST images for each target class
based purely on model predictions (ignores filenames — HRIPCB filenames
do NOT match GT class labels).

For each target class, saves the top TOP_K images (annotated) to
selected_final/<class_name>/ so you can visually inspect and pick.

Run from project root:
    python scripts/select_strict_qualitative.py
"""

import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")  # CPU only

from pathlib import Path
from ultralytics import YOLO

DATASET_PATH = Path("datasets/HRIPCB/test/images")
OUTPUT_DIR   = Path("selected_final")
CONF_THRESHOLD = 0.5
TOP_K = 5  # save 5 candidates per class so you can pick the clearest

# Only show these 3 classes in the figure — change as needed
TARGET_CLASSES = ["missing_hole", "mouse_bite", "short"]

MODELS = {
    "baseline": "runs/experiments/baseline/yolov8n/weights/best.pt",
    "teacher":  "runs/experiments/teacher/yolov8m/weights/best.pt",
    "kd":       "runs/experiments/kd/kd_combined/weights/best.pt",
}

# ── load models ──────────────────────────────────────────────────────────────
print("Loading models...")
loaded = {k: YOLO(v) for k, v in MODELS.items()}
class_names = loaded["kd"].names  # {0: 'missing_hole', ...}

# ── helper ───────────────────────────────────────────────────────────────────
def best_conf_for_class(model, img_path, target_cls_name):
    """Return the highest confidence detection of target_cls_name, or 0."""
    result = model.predict(str(img_path), imgsz=640, conf=0.001, verbose=False)[0]
    if result.boxes is None or len(result.boxes) == 0:
        return 0.0
    best = 0.0
    for box in result.boxes:
        if class_names[int(box.cls[0])] == target_cls_name:
            best = max(best, float(box.conf[0]))
    return best

# ── scan ─────────────────────────────────────────────────────────────────────
print("Scanning test images...")

# candidates[cls] = list of (min_conf_across_models, img_path)
candidates = {cls: [] for cls in TARGET_CLASSES}

all_images = sorted(DATASET_PATH.glob("*.jpg"))
total = len(all_images)

for i, img_path in enumerate(all_images):
    if i % 100 == 0:
        print(f"  {i}/{total} ...")

    for cls_name in TARGET_CLASSES:
        # KD must detect it
        kd_conf = best_conf_for_class(loaded["kd"], img_path, cls_name)
        if kd_conf < CONF_THRESHOLD:
            continue

        # Baseline and teacher should also detect it
        bl_conf  = best_conf_for_class(loaded["baseline"], img_path, cls_name)
        tea_conf = best_conf_for_class(loaded["teacher"],  img_path, cls_name)

        if bl_conf < CONF_THRESHOLD or tea_conf < CONF_THRESHOLD:
            continue  # all 3 must agree

        min_conf = min(kd_conf, bl_conf, tea_conf)
        candidates[cls_name].append((min_conf, img_path))

# ── save top-K annotated images per class ────────────────────────────────────
print("\nSaving top candidates...")

for cls_name in TARGET_CLASSES:
    items = sorted(candidates[cls_name], key=lambda x: -x[0])[:TOP_K]
    if not items:
        print(f"  [!] No candidates found for: {cls_name}")
        continue

    save_dir = OUTPUT_DIR / cls_name
    save_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n  {cls_name}: {len(items)} candidates")
    for rank, (conf, img_path) in enumerate(items):
        print(f"    #{rank+1}  min_conf={conf:.3f}  {img_path.name}")

        # Save the KD-annotated version so you can see the boxes
        result = loaded["kd"].predict(str(img_path), imgsz=640, conf=0.25, verbose=False)[0]
        annotated = result.plot()

        import cv2
        out_path = save_dir / f"rank{rank+1}_{img_path.name}"
        cv2.imwrite(str(out_path), annotated)

print(f"\nDone. Open selected_final/<class>/ to visually inspect.")
print("Pick the best image for each class and put the path into")
print("generate_qualitative_figure.py → IMAGES list.")
