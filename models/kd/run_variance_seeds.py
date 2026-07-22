"""
run_variance_seeds.py — Repeated-seed runs for the standard-deviation
revision requested by a reviewer.

Trains Student baseline and Combined KD + P3-heavy at seeds 43 and 44
(seed 42 results already exist from the original ablation study under
runs/experiments/baseline/yolov8n and runs/experiments/kd/kd_combined_p3_heavy).
Runs sequentially on a single GPU. Outputs go to new, distinctly named
directories so the original seed-42 runs are never touched.

Usage (from project root):
    python models/kd/run_variance_seeds.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ultralytics import YOLO
from kd_trainer import KDTrainer

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
TEACHER_PATH = os.path.join(PROJECT_ROOT, "runs", "experiments", "teacher", "yolov8m", "weights", "best.pt")
DATA_YAML    = os.path.join(PROJECT_ROOT, "data", "pcb_defect.yaml")

EPOCHS, BATCH_SIZE, IMAGE_SIZE, DEVICE = 200, 16, 640, "0"

SHARED_OVERRIDES = dict(
    data=DATA_YAML,
    epochs=EPOCHS,
    batch=BATCH_SIZE,
    imgsz=IMAGE_SIZE,
    device=DEVICE,
    workers=8,
    patience=50,
    optimizer="SGD",
    lr0=0.01,
    lrf=0.1,
    momentum=0.937,
    weight_decay=0.0005,
    cos_lr=True,
    warmup_epochs=3,
    warmup_momentum=0.8,
    warmup_bias_lr=0.1,
    box=7.5,
    cls=0.5,
    dfl=1.5,
    hsv_h=0.015,
    hsv_s=0.7,
    hsv_v=0.4,
    fliplr=0.5,
    mosaic=0.5,
    mixup=0.1,
    val=True,
    plots=True,
    save_json=True,
    save=True,
    save_period=20,
    amp=True,
    cache=False,
    verbose=True,
    exist_ok=False,
    deterministic=False,
)

BASELINE_PROJECT = os.path.join(PROJECT_ROOT, "runs", "experiments", "baseline_variance")
KD_PROJECT       = os.path.join(PROJECT_ROOT, "runs", "experiments", "kd_variance")

SEEDS = [43, 44]


def run_baseline(seed):
    name = f"yolov8n_seed{seed}"
    print(f"\n{'='*65}\n  Baseline run, seed={seed}\n{'='*65}")
    model = YOLO("yolov8n.pt")
    model.train(
        project=BASELINE_PROJECT,
        name=name,
        seed=seed,
        **SHARED_OVERRIDES,
    )


def run_kd_p3_heavy(seed):
    name = f"kd_combined_p3_heavy_seed{seed}"
    print(f"\n{'='*65}\n  Combined KD + P3-heavy run, seed={seed}\n{'='*65}")
    trainer = KDTrainer(
        teacher_path=TEACHER_PATH,
        alpha=0.7, beta=0.2, gamma=0.1, delta=0.0, temperature=4.0,
        feat_scale_weights=[0.6, 0.3, 0.1],
        overrides=dict(
            model="yolov8n.pt",
            project=KD_PROJECT,
            name=name,
            seed=seed,
            **SHARED_OVERRIDES,
        ),
    )
    trainer.train()


def main():
    if not os.path.isfile(TEACHER_PATH):
        print(f"[ERROR] Teacher not found: {TEACHER_PATH}")
        sys.exit(1)
    if not os.path.isfile(DATA_YAML):
        print(f"[ERROR] Dataset YAML not found: {DATA_YAML}")
        sys.exit(1)

    for seed in SEEDS:
        run_baseline(seed)
    for seed in SEEDS:
        run_kd_p3_heavy(seed)

    print("\n" + "=" * 65)
    print("  All 4 variance runs complete.")
    print(f"  Baseline weights : {BASELINE_PROJECT}/yolov8n_seed<43|44>/weights/best.pt")
    print(f"  KD weights       : {KD_PROJECT}/kd_combined_p3_heavy_seed<43|44>/weights/best.pt")
    print("=" * 65)


if __name__ == "__main__":
    main()
