"""
train_kd.py — Knowledge Distillation Training Script
=====================================================
This is the MAIN script to train the YOLOv8n student model using knowledge
distillation from the pre-trained YOLOv8m teacher model.

HOW TO RUN:
-----------
    python train_kd.py

BEFORE RUNNING:
---------------
  1. Set TEACHER_PATH to your trained YOLOv8m checkpoint (the teacher).
  2. Set DATA_YAML to the path of your pcb_defect.yaml file (or leave as default).
  3. (Optional) Adjust batch size, epochs, or KD hyperparameters below.

OUTPUT:
-------
  - Best student model saved to: runs/kd/kd_training/weights/best.pt
  - Last checkpoint saved to:    runs/kd/kd_training/weights/last.pt
  - Training metrics CSV:        runs/kd/kd_training/results.csv
  - Training plots:              runs/kd/kd_training/*.png
"""

import os
import sys

# Make sure we can import from the current directory
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from kd_trainer import KDTrainer


# ==============================================================================
#  CONFIGURATION — Edit these values before running
# ==============================================================================

# Path to your trained YOLOv8m teacher checkpoint
# This is the model that was already fine-tuned on the PCB dataset.
TEACHER_PATH = "path/to/your/yolov8m_pcb_best.pt"   # <-- CHANGE THIS

# Path to dataset configuration file
DATA_YAML = "data/pcb_defect.yaml"   # <-- Make sure path is set inside this file

# Student model (pretrained on COCO as starting point)
# Use "yolov8n.pt" for nano, "yolov8s.pt" for small, or a custom checkpoint path
STUDENT_MODEL = "yolov8n.pt"

# Training duration
EPOCHS = 100           # Number of training epochs

# Hardware settings
BATCH_SIZE = 8         # Images per batch. Use 16 if you have enough GPU memory (>12GB).
IMAGE_SIZE = 640       # Input image size in pixels (must match teacher training)
DEVICE = "0"           # GPU device. Use "0" for first GPU, "cpu" for CPU (slow).

# ----------------------------------------------------------------------------
#  Knowledge Distillation Hyperparameters
#  (These are the recommended defaults from the implementation plan)
# ----------------------------------------------------------------------------

# ALPHA: Weight for the standard YOLO detection loss (box + class + DFL)
# Higher alpha → student focuses more on ground truth labels
ALPHA = 0.7

# BETA: Weight for the logit KD loss (KL divergence between student & teacher predictions)
# Higher beta → student tries harder to match teacher's prediction confidence
BETA = 0.2

# GAMMA: Weight for the feature KD loss (MSE between intermediate feature maps)
# Higher gamma → student tries harder to mimic teacher's internal representations
# Set to 0.0 to disable feature KD and use logit KD only
GAMMA = 0.1

# TEMPERATURE: Softens the teacher's prediction distribution
# Higher T → softer probabilities → more "dark knowledge" transferred
# Recommended range: 2–6. Default: 4.0
TEMPERATURE = 4.0

# Note: alpha + beta + gamma should sum to approximately 1.0
# Default: 0.7 + 0.2 + 0.1 = 1.0 ✓

# ==============================================================================
#  TRAINING — Do not change anything below unless you know what you're doing
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
        print("  Please set TEACHER_PATH to the path of your trained YOLOv8m model.")
        print("  Example: TEACHER_PATH = '/home/user/runs/detect/train/weights/best.pt'")
        sys.exit(1)

    # Validate dataset YAML
    if not os.path.isfile(DATA_YAML):
        print(f"\n[ERROR] Dataset YAML not found: {DATA_YAML}")
        print("  Please ensure data/pcb_defect.yaml exists and its 'path' is set correctly.")
        sys.exit(1)

    print("\nStarting training...\n")

    trainer = KDTrainer(
        teacher_path=TEACHER_PATH,
        alpha=ALPHA,
        beta=BETA,
        gamma=GAMMA,
        temperature=TEMPERATURE,
        overrides=dict(
            model=STUDENT_MODEL,      # Student model (pretrained on COCO as starting point)
            data=DATA_YAML,
            epochs=EPOCHS,
            batch=BATCH_SIZE,
            imgsz=IMAGE_SIZE,
            device=DEVICE,
            project="runs/kd",
            name="kd_training",
            exist_ok=True,           # Overwrite existing run folder
            verbose=True,
        )
    )

    trainer.train()

    print("\n" + "=" * 65)
    print("  Training Complete!")
    print("=" * 65)
    print(f"  Best model  : runs/kd/kd_training/weights/best.pt")
    print(f"  Last model  : runs/kd/kd_training/weights/last.pt")
    print(f"  Metrics CSV : runs/kd/kd_training/results.csv")
    print("\nTo evaluate your distilled model, run:")
    print("  python evaluate.py --weights runs/kd/kd_training/weights/best.pt")
    print("=" * 65)


if __name__ == "__main__":
    main()