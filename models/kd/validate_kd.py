"""
validate_kd.py — Validation Script for KD Experiments
======================================================
Runs validation on a trained KD model and saves results under
runs/experiments/kd_val/<EXPERIMENT_NAME>/ alongside training results.

Usage (from project root):
    python models/kd/validate_kd.py
"""

import os
import sys
from ultralytics import YOLO

# ==============================================================================
#  CONFIGURATION — edit these before running
# ==============================================================================

# Path to the trained weights
WEIGHTS = "runs/experiments/kd/kd_combined_p3_dominant/weights/best.pt"

# Must match the experiment folder name under runs/experiments/kd/
EXPERIMENT_NAME = "kd_combined_p3_dominant"

DATA_YAML  = "data/pcb_defect.yaml"
IMAGE_SIZE = 640
DEVICE     = "0"
BATCH_SIZE = 16

# Absolute path to keep Ultralytics from prepending runs/detect/
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
VAL_PROJECT  = os.path.join(PROJECT_ROOT, "runs", "experiments", "kd_val")


# ==============================================================================
#  VALIDATION
# ==============================================================================

def main():
    save_dir = os.path.join(VAL_PROJECT, EXPERIMENT_NAME)

    print("=" * 65)
    print("  KD Model Validation")
    print("=" * 65)
    print(f"  Weights       : {WEIGHTS}")
    print(f"  Save to       : {save_dir}/")
    print(f"  Dataset       : {DATA_YAML}")
    print("=" * 65)

    if not os.path.isfile(WEIGHTS):
        print(f"\n[ERROR] Weights not found: {WEIGHTS}")
        sys.exit(1)

    if not os.path.isfile(DATA_YAML):
        print(f"\n[ERROR] Dataset YAML not found: {DATA_YAML}")
        sys.exit(1)

    model = YOLO(WEIGHTS)

    metrics = model.val(
        data=DATA_YAML,
        imgsz=IMAGE_SIZE,
        batch=BATCH_SIZE,
        device=DEVICE,
        split="val",
        project=VAL_PROJECT,   # absolute path — no runs/detect/ prepended
        name=EXPERIMENT_NAME,
        exist_ok=True,
        verbose=True,
        save_json=False,
        plots=True,
    )

    # ------------------------------------------------------------------
    # Speed / FPS
    # metrics.speed = {'preprocess': ms, 'inference': ms, 'loss': ms, 'postprocess': ms}
    # FPS is based on inference + pre/post-processing (end-to-end per image)
    # ------------------------------------------------------------------
    spd            = metrics.speed
    preprocess_ms  = spd.get("preprocess",  0.0)
    inference_ms   = spd.get("inference",   0.0)
    postprocess_ms = spd.get("postprocess", 0.0)
    total_ms       = preprocess_ms + inference_ms + postprocess_ms
    fps            = 1000.0 / total_ms if total_ms > 0 else 0.0

    # ------------------------------------------------------------------
    # Save numeric results to a text file for easy tracking
    # ------------------------------------------------------------------
    class_names = model.names  # {0: 'missing_hole', ...}
    os.makedirs(save_dir, exist_ok=True)
    result_path = os.path.join(save_dir, "val_results.txt")

    with open(result_path, "w") as f:
        f.write(f"Experiment : {EXPERIMENT_NAME}\n")
        f.write(f"Weights    : {WEIGHTS}\n")
        f.write(f"Split      : test\n")
        f.write("=" * 50 + "\n")
        f.write(f"{'Metric':<20} {'Value':>10}\n")
        f.write("-" * 32 + "\n")
        f.write(f"{'mAP50':<20} {metrics.box.map50:>10.4f}\n")
        f.write(f"{'mAP50-95':<20} {metrics.box.map:>10.4f}\n")
        f.write(f"{'Precision':<20} {metrics.box.mp:>10.4f}\n")
        f.write(f"{'Recall':<20} {metrics.box.mr:>10.4f}\n")
        f.write("=" * 50 + "\n")
        f.write(f"\n{'Speed (ms/image)'}\n")
        f.write("-" * 32 + "\n")
        f.write(f"  {'Preprocess':<20} {preprocess_ms:.2f} ms\n")
        f.write(f"  {'Inference':<20} {inference_ms:.2f} ms\n")
        f.write(f"  {'Postprocess':<20} {postprocess_ms:.2f} ms\n")
        f.write(f"  {'Total (end-to-end)':<20} {total_ms:.2f} ms\n")
        f.write(f"  {'FPS':<20} {fps:.1f}\n")
        f.write("=" * 50 + "\n")
        f.write(f"\n{'Per-class mAP50-95'}\n")
        f.write("-" * 32 + "\n")
        for i, ap in enumerate(metrics.box.maps):
            name = class_names.get(i, str(i))
            f.write(f"  {name:<20} {ap:.4f}\n")

    print("\n" + "=" * 65)
    print("  Validation Results")
    print("=" * 65)
    print(f"  mAP50     : {metrics.box.map50:.4f}")
    print(f"  mAP50-95  : {metrics.box.map:.4f}")
    print(f"  Precision : {metrics.box.mp:.4f}")
    print(f"  Recall    : {metrics.box.mr:.4f}")
    print(f"\n  Speed (ms/image)")
    print(f"  Preprocess  : {preprocess_ms:.2f} ms")
    print(f"  Inference   : {inference_ms:.2f} ms")
    print(f"  Postprocess : {postprocess_ms:.2f} ms")
    print(f"  FPS         : {fps:.1f}")
    print(f"\n  Results saved to : {save_dir}/")
    print(f"  Summary file     : {result_path}")
    print("=" * 65)


if __name__ == "__main__":
    main()
