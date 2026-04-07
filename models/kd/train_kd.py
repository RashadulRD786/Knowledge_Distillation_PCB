"""
train_kd.py — Knowledge Distillation Training Script
=====================================================
This is the MAIN script to train the YOLOv8n student model using knowledge
distillation from the pre-trained YOLOv8m teacher model.
"""

import os
import sys

# Make sure we can import from the current directory
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from kd_trainer import KDTrainer


# ==============================================================================
#  CONFIGURATION
# ==============================================================================

TEACHER_PATH = "runs/experiments/teacher/yolov8m/weights/best.pt"
DATA_YAML = "data/pcb_defect.yaml"
STUDENT_MODEL = "yolov8n.pt"

# ✅ MATCH BASELINE PROTOCOL
EPOCHS = 200
BATCH_SIZE = 16
IMAGE_SIZE = 640
DEVICE = "0"

# ----------------------------------------------------------------------------
#  Knowledge Distillation Hyperparameters (Logits KD FIRST)
# ----------------------------------------------------------------------------
ALPHA = 0.8
BETA = 0.2
GAMMA = 0.0      # 🔴 Logits KD only (IMPORTANT)
TEMPERATURE = 4.0


# ==============================================================================
#  TRAINING
# ==============================================================================

def main():
    print("=" * 65)
    print("  YOLOv8 Knowledge Distillation Training")
    print("=" * 65)
    print(f"  Teacher model : {TEACHER_PATH}")
    print(f"  Dataset       : {DATA_YAML}")
    print(f"  Epochs        : {EPOCHS}")
    print(f"  Batch size    : {BATCH_SIZE}")
    print(f"  Image size    : {IMAGE_SIZE}")
    print(f"  Device        : GPU {DEVICE}")
    print(f"  Alpha (det)   : {ALPHA}")
    print(f"  Beta  (logit) : {BETA}")
    print(f"  Gamma (feat)  : {GAMMA}")
    print(f"  Temperature   : {TEMPERATURE}")
    print("=" * 65)

    # Validate teacher path
    if not os.path.isfile(TEACHER_PATH):
        print(f"\n[ERROR] Teacher checkpoint not found: {TEACHER_PATH}")
        sys.exit(1)

    # Validate dataset YAML
    if not os.path.isfile(DATA_YAML):
        print(f"\n[ERROR] Dataset YAML not found: {DATA_YAML}")
        sys.exit(1)

    print("\nStarting training...\n")

    trainer = KDTrainer(
        teacher_path=TEACHER_PATH,
        alpha=ALPHA,
        beta=BETA,
        gamma=GAMMA,
        temperature=TEMPERATURE,
        overrides=dict(

            # -------------------------
            # Core (MATCH BASELINE)
            # -------------------------
            model=STUDENT_MODEL,
            data=DATA_YAML,
            epochs=EPOCHS,
            batch=BATCH_SIZE,
            imgsz=IMAGE_SIZE,
            device=DEVICE,
            workers=8,
            patience=50,

            # -------------------------
            # Optimization (CRITICAL)
            # -------------------------
            optimizer="SGD",
            lr0=0.01,
            lrf=0.1,
            momentum=0.937,
            weight_decay=0.0005,
            cos_lr=True,

            # -------------------------
            # Warmup
            # -------------------------
            warmup_epochs=3,
            warmup_momentum=0.8,
            warmup_bias_lr=0.1,

            # -------------------------
            # Loss Weights
            # -------------------------
            box=7.5,
            cls=0.5,
            dfl=1.5,

            # -------------------------
            # Augmentation (MATCH BASELINE)
            # -------------------------
            hsv_h=0.015,
            hsv_s=0.7,
            hsv_v=0.4,
            fliplr=0.5,
            mosaic=0.5,
            mixup=0.1,

            # -------------------------
            # Logging
            # -------------------------
            project="../experiments/kd",
            name="kd_logit_only",
            exist_ok=False,
            verbose=True,

            # -------------------------
            # Performance
            # -------------------------
            amp=True,
            cache=False,

            # -------------------------
            # Reproducibility
            # -------------------------
            seed=42,
            deterministic=False,
        )
    )

    trainer.train()

    print("\n" + "=" * 65)
    print("  Training Complete!")
    print("=" * 65)
    print(f"  Best model  : runs/kd/kd_training/weights/best.pt")
    print(f"  Last model  : runs/kd/kd_training/weights/last.pt")
    print(f"  Metrics CSV : runs/kd/kd_training/results.csv")
    print("=" * 65)


if __name__ == "__main__":
    main()