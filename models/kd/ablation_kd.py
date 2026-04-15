"""
ablation_kd.py — Ablation Study Runner for KD Experiments
==========================================================
All 5 ablation configurations are defined here.
To run an experiment, set RUN to the desired experiment name and execute:

    python models/kd/ablation_kd.py

Run one experiment at a time. After each finishes, run validate_kd.py
(with EXPERIMENT_NAME and WEIGHTS updated) to get test results.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from kd_trainer import KDTrainer

# ==============================================================================
#  SELECT EXPERIMENT — change this line only
# ==============================================================================

RUN = "kd_combined_p3_dominant"   # options: see CONFIGS below

# ==============================================================================
#  SHARED SETTINGS
# ==============================================================================

PROJECT_ROOT  = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
TEACHER_PATH  = os.path.join(PROJECT_ROOT, "runs", "experiments", "teacher", "yolov8m", "weights", "best.pt")
DATA_YAML     = os.path.join(PROJECT_ROOT, "data", "pcb_defect.yaml")
KD_PROJECT    = os.path.join(PROJECT_ROOT, "runs", "experiments", "kd")
STUDENT_MODEL         = "yolov8n.pt"
STUDENT_MODEL_SCRATCH = "yolov8n.yaml"   # architecture only, no pretrained weights

EPOCHS     = 200
BATCH_SIZE = 16
IMAGE_SIZE = 640
DEVICE     = "0"

# ==============================================================================
#  ABLATION CONFIGURATIONS
# ==============================================================================

CONFIGS = {
    "kd_logit_strong": dict(
        alpha=0.7, beta=0.3, gamma=0.0, delta=0.0, temperature=4.0,
        description="Logits KD (stronger beta=0.3, T=4)",
    ),
    "kd_logit_T2": dict(
        alpha=0.7, beta=0.3, gamma=0.0, delta=0.0, temperature=2.0,
        description="Logits KD (T=2 — sharper targets)",
    ),
    "kd_logit_T6": dict(
        alpha=0.7, beta=0.3, gamma=0.0, delta=0.0, temperature=6.0,
        description="Logits KD (T=6 — softer targets)",
    ),
    "kd_feat_only": dict(
        alpha=0.8, beta=0.0, gamma=0.2, delta=0.0, temperature=4.0,
        description="Feature KD only (neck P3/P4/P5 MSE)",
    ),
    "kd_combined": dict(
        alpha=0.7, beta=0.2, gamma=0.1, delta=0.0, temperature=4.0,
        description="Combined logit + feature KD",
    ),
    "kd_box": dict(
        alpha=0.8, beta=0.0, gamma=0.0, delta=0.2, temperature=4.0,
        description="Box regression distillation only (DFL outputs)",
    ),
    "kd_combined_full": dict(
        alpha=0.7, beta=0.1, gamma=0.1, delta=0.1, temperature=4.0,
        description="Combined logit + feature + box KD",
    ),

    # ------------------------------------------------------------------
    #  FROM-SCRATCH EXPERIMENTS
    #  Student initialized from random weights (no COCO pretraining)
    # ------------------------------------------------------------------
    "scratch_baseline": dict(
        alpha=1.0, beta=0.0, gamma=0.0, delta=0.0, temperature=4.0,
        description="YOLOv8n from scratch — no KD (baseline for scratch comparison)",
        scratch=True,
    ),
    "scratch_kd_combined": dict(
        alpha=0.7, beta=0.2, gamma=0.1, delta=0.0, temperature=4.0,
        description="YOLOv8n from scratch + combined KD (logit + feature)",
        scratch=True,
    ),

    # ------------------------------------------------------------------
    #  P3-WEIGHTED FEATURE DISTILLATION EXPERIMENTS
    #  Hypothesis: upweighting P3 (finest-grained FPN scale) improves
    #  AP_small by forcing the student to better mimic the teacher's
    #  small-object spatial features.
    # ------------------------------------------------------------------
    "kd_combined_p3_heavy": dict(
        alpha=0.7, beta=0.2, gamma=0.1, delta=0.0, temperature=4.0,
        feat_scale_weights=[0.6, 0.3, 0.1],
        description="Combined KD — P3-heavy feature weights [0.6, 0.3, 0.1]",
    ),
    "kd_combined_p3_dominant": dict(
        alpha=0.7, beta=0.2, gamma=0.1, delta=0.0, temperature=4.0,
        feat_scale_weights=[0.7, 0.2, 0.1],
        description="Combined KD — P3-dominant feature weights [0.7, 0.2, 0.1]",
    ),
}

# ==============================================================================
#  TRAINING
# ==============================================================================

def main():
    if RUN not in CONFIGS:
        print(f"[ERROR] Unknown experiment '{RUN}'. Choose from: {list(CONFIGS.keys())}")
        sys.exit(1)

    cfg = CONFIGS[RUN]
    is_scratch = cfg.get("scratch", False)
    student_model = STUDENT_MODEL_SCRATCH if is_scratch else STUDENT_MODEL

    print("=" * 65)
    print(f"  Ablation: {cfg['description']}")
    print("=" * 65)
    print(f"  Name        : {RUN}")
    print(f"  Teacher     : {TEACHER_PATH}")
    print(f"  Student     : {student_model} ({'scratch' if is_scratch else 'pretrained'})")
    print(f"  Alpha            : {cfg['alpha']}")
    print(f"  Beta             : {cfg['beta']}")
    print(f"  Gamma            : {cfg['gamma']}")
    print(f"  Delta            : {cfg['delta']}")
    print(f"  Temperature      : {cfg['temperature']}")
    _fw = cfg.get("feat_scale_weights")
    print(f"  Feat scale wts   : {_fw if _fw else 'equal (1/3 each)'}")
    print("=" * 65)

    if not os.path.isfile(TEACHER_PATH):
        print(f"\n[ERROR] Teacher not found: {TEACHER_PATH}")
        sys.exit(1)

    if not os.path.isfile(DATA_YAML):
        print(f"\n[ERROR] Dataset YAML not found: {DATA_YAML}")
        sys.exit(1)

    trainer = KDTrainer(
        teacher_path=TEACHER_PATH,
        alpha=cfg["alpha"],
        beta=cfg["beta"],
        gamma=cfg["gamma"],
        delta=cfg["delta"],
        temperature=cfg["temperature"],
        feat_scale_weights=cfg.get("feat_scale_weights", None),
        overrides=dict(
            # Core
            model=student_model,
            data=DATA_YAML,
            epochs=EPOCHS,
            batch=BATCH_SIZE,
            imgsz=IMAGE_SIZE,
            device=DEVICE,
            workers=8,
            patience=50,

            # Optimisation
            optimizer="SGD",
            lr0=0.01,
            lrf=0.1,
            momentum=0.937,
            weight_decay=0.0005,
            cos_lr=True,

            # Warmup
            warmup_epochs=3,
            warmup_momentum=0.8,
            warmup_bias_lr=0.1,

            # Loss weights
            box=7.5,
            cls=0.5,
            dfl=1.5,

            # Augmentation
            hsv_h=0.015,
            hsv_s=0.7,
            hsv_v=0.4,
            fliplr=0.5,
            mosaic=0.5,
            mixup=0.1,

            # Logging
            project=KD_PROJECT,   # absolute path — prevents runs/detect/ prepending
            name=RUN,
            exist_ok=False,
            verbose=True,

            # Performance
            amp=True,
            cache=False,

            # Reproducibility
            seed=42,
            deterministic=False,
        )
    )

    trainer.train()

    print("\n" + "=" * 65)
    print("  Training Complete!")
    print("=" * 65)
    print(f"  Best weights : {KD_PROJECT}/{RUN}/weights/best.pt")
    print(f"  Next step    : update WEIGHTS and EXPERIMENT_NAME in")
    print(f"                 validate_kd.py and run validation.")
    print("=" * 65)


if __name__ == "__main__":
    main()
