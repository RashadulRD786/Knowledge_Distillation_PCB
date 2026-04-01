"""
run_ablation.py — Ablation Study Runner
========================================
Runs 7 ablation experiments and saves results to ablation_results.csv.

EXPERIMENTS:
  E1  — Baseline YOLOv8n (no knowledge distillation)
  E2  — Logit KD only (KL divergence, no feature loss)
  E3  — Feature KD only (MSE on features, no logit loss)
  E4  — Combined KD (logit + feature, T=4) ← recommended default / reference point
  E5a — Temperature sweep: T=1 (no softening — hard targets)
  E5b — Temperature sweep: T=2 (light softening)
  E5d — Temperature sweep: T=8 (very soft — may lose signal)
        NOTE: T=4 result is in E4 (identical config). Use E4 as the T=4 data point
        when plotting the full temperature sweep (T=1,2,4,8).
  E6a — Alpha=0.9 (trust ground truth labels more, less KD influence)
  E6c — Alpha=0.5 (equal weight for detection loss and KD)
        NOTE: Alpha=0.7 result is in E4 (identical config). Use E4 as the α=0.7
        data point when comparing the alpha sweep.
        IMPORTANT: Run E5 first. Set E6_BEST_TEMPERATURE (below) to the best T
        from E5 before running E6a and E6c.
  E7b — Teacher = YOLOv8l  (requires v8l PCB checkpoint)
  E7c — Teacher = YOLOv8x  (requires v8x PCB checkpoint)
        NOTE: Teacher=YOLOv8m result is in E4 (identical config). Use E4 as the
        v8m baseline when comparing teacher sizes.

HOW TO RUN:
-----------
    # Recommended order — run E1–E5 first, then set E6_BEST_TEMPERATURE below:
    python run_ablation.py --exp E1 E2 E3 E4 E5a E5b E5d

    # Then update E6_BEST_TEMPERATURE in this file, and run E6:
    python run_ablation.py --exp E6a E6c

    # Run all default experiments at once (skips E7b/E7c which need extra checkpoints):
    python run_ablation.py

    # Run teacher size comparison (requires v8l and v8x PCB checkpoints):
    python run_ablation.py --exp E7b E7c

OUTPUT:
-------
    ablation_results.csv   — table of all results (appended after each experiment)
    runs/ablation/         — one subfolder per experiment with model checkpoints

NOTE on E7 (teacher size experiments):
---------------------------------------
E7b and E7c require YOLOv8l and YOLOv8x checkpoints already fine-tuned on the
PCB dataset. If you only have YOLOv8m, these experiments cannot be run.
Compare E7b/E7c results against E4 (which used YOLOv8m as teacher).
"""

import os
import sys
import csv
import time
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
from ultralytics import YOLO
from kd_trainer import KDTrainer


# ==============================================================================
#  CONFIGURATION — Edit these paths before running
# ==============================================================================

DATA_YAML = "data/pcb_defect.yaml"   # <-- Make sure path inside this file is correct

# Teacher checkpoint paths
TEACHER_V8M = "path/to/your/yolov8m_pcb_best.pt"   # <-- REQUIRED: change this
TEACHER_V8L = "path/to/your/yolov8l_pcb_best.pt"   # Optional: only needed for E7b
TEACHER_V8X = "path/to/your/yolov8x_pcb_best.pt"   # Optional: only needed for E7c

# Training settings (shared across all experiments)
EPOCHS = 100
BATCH_SIZE = 8
IMAGE_SIZE = 640
DEVICE = "0"

RESULTS_CSV = "ablation_results.csv"

# ── E6 temperature setting ─────────────────────────────────────────────────
# IMPORTANT: After E5 finishes, look at ablation_results.csv and find which
# temperature (T=1, 2, 4, or 8) gave the best mAP50-95. Then set this variable
# to that value before running E6a and E6c.
# Example: if T=2 was best in E5, set E6_BEST_TEMPERATURE = 2.0
# Default: 4.0 (same as E4 — a safe starting point before E5 results are known)
E6_BEST_TEMPERATURE = 4.0   # <-- UPDATE THIS after running E5

# ==============================================================================
#  EXPERIMENT DEFINITIONS
# ==============================================================================
# Each experiment is a dict with:
#   name        : unique experiment identifier (used for folder name)
#   description : human-readable description
#   mode        : "baseline" or "kd"
#   teacher     : path to teacher checkpoint (None for baseline)
#   alpha       : detection loss weight
#   beta        : logit KD loss weight
#   gamma       : feature KD loss weight
#   temperature : KD temperature

EXPERIMENTS = {

    # ── E1: Baseline ──────────────────────────────────────────────────────────
    "E1": {
        "name": "E1_baseline",
        "description": "YOLOv8n trained without knowledge distillation",
        "mode": "baseline",
        "teacher": None,
        "alpha": 1.0, "beta": 0.0, "gamma": 0.0, "temperature": 4.0,
    },

    # ── E2: Logit KD only ─────────────────────────────────────────────────────
    "E2": {
        "name": "E2_logit_kd",
        "description": "Logit KD only (KL divergence, gamma=0, beta absorbs gamma weight)",
        "mode": "kd",
        "teacher": TEACHER_V8M,
        "alpha": 0.7, "beta": 0.3, "gamma": 0.0, "temperature": 4.0,
    },

    # ── E3: Feature KD only ───────────────────────────────────────────────────
    "E3": {
        "name": "E3_feature_kd",
        "description": "Feature KD only (MSE on features, beta=0, gamma absorbs beta weight)",
        "mode": "kd",
        "teacher": TEACHER_V8M,
        "alpha": 0.7, "beta": 0.0, "gamma": 0.3, "temperature": 4.0,
    },

    # ── E4: Combined KD ───────────────────────────────────────────────────────
    "E4": {
        "name": "E4_combined_kd",
        "description": "Combined KD: logit + feature (default recommended config)",
        "mode": "kd",
        "teacher": TEACHER_V8M,
        "alpha": 0.7, "beta": 0.2, "gamma": 0.1, "temperature": 4.0,
    },

    # ── E5: Temperature sweep ─────────────────────────────────────────────────
    "E5a": {
        "name": "E5_T1",
        "description": "Temperature sweep: T=1 (no softening — hard targets)",
        "mode": "kd",
        "teacher": TEACHER_V8M,
        "alpha": 0.7, "beta": 0.2, "gamma": 0.1, "temperature": 1.0,
    },
    "E5b": {
        "name": "E5_T2",
        "description": "Temperature sweep: T=2 (light softening)",
        "mode": "kd",
        "teacher": TEACHER_V8M,
        "alpha": 0.7, "beta": 0.2, "gamma": 0.1, "temperature": 2.0,
    },
    # E5c (T=4) is intentionally absent — it is identical to E4.
    # Use E4 results as the T=4 data point in the temperature sweep plot.
    "E5d": {
        "name": "E5_T8",
        "description": "Temperature sweep: T=8 (very soft — may lose signal)",
        "mode": "kd",
        "teacher": TEACHER_V8M,
        "alpha": 0.7, "beta": 0.2, "gamma": 0.1, "temperature": 8.0,
    },

    # ── E6: Alpha sweep ───────────────────────────────────────────────────────
    # Beta and gamma scale proportionally, keeping their 2:1 ratio.
    # Uses E6_BEST_TEMPERATURE (set at the top of this file after E5 results are in).
    # E6b (alpha=0.7) is intentionally absent — it is identical to E4.
    # Use E4 results as the alpha=0.7 data point in the alpha sweep plot.
    "E6a": {
        "name": "E6_alpha_0.9",
        "description": "Alpha=0.9: trust ground truth labels more",
        "mode": "kd",
        "teacher": TEACHER_V8M,
        "alpha": 0.9, "beta": 0.067, "gamma": 0.033, "temperature": E6_BEST_TEMPERATURE,
    },
    "E6c": {
        "name": "E6_alpha_0.5",
        "description": "Alpha=0.5: equal weight to detection and KD",
        "mode": "kd",
        "teacher": TEACHER_V8M,
        "alpha": 0.5, "beta": 0.333, "gamma": 0.167, "temperature": E6_BEST_TEMPERATURE,
    },

    # ── E7: Teacher size ──────────────────────────────────────────────────────
    # E7a (teacher=YOLOv8m) is intentionally absent — it is identical to E4.
    # Use E4 results as the YOLOv8m baseline when comparing teacher sizes.
    "E7b": {
        "name": "E7_teacher_v8l",
        "description": "Teacher size: YOLOv8l (43.6M params) — requires v8l checkpoint",
        "mode": "kd",
        "teacher": TEACHER_V8L,
        "alpha": 0.7, "beta": 0.2, "gamma": 0.1, "temperature": 4.0,
    },
    "E7c": {
        "name": "E7_teacher_v8x",
        "description": "Teacher size: YOLOv8x (68.1M params) — requires v8x checkpoint",
        "mode": "kd",
        "teacher": TEACHER_V8X,
        "alpha": 0.7, "beta": 0.2, "gamma": 0.1, "temperature": 4.0,
    },
}


# ==============================================================================
#  RESULTS LOGGING
# ==============================================================================

def init_results_csv():
    """Create the results CSV file with headers if it doesn't exist."""
    if not os.path.isfile(RESULTS_CSV):
        with open(RESULTS_CSV, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'experiment', 'description', 'alpha', 'beta', 'gamma', 'temperature',
                'teacher', 'mAP50', 'mAP50_95', 'precision', 'recall',
                'fps', 'model_size_mb', 'weights_path', 'training_time_hrs'
            ])
    print(f"Results will be saved to: {RESULTS_CSV}")


def append_result(exp_id: str, exp_cfg: dict, metrics: dict, weights_path: str, training_time_hrs: float):
    """Append one experiment result row to the CSV, replacing any existing row for the same experiment."""
    # Remove existing row for this experiment (handles re-runs)
    if os.path.isfile(RESULTS_CSV):
        with open(RESULTS_CSV, 'r', newline='') as f:
            reader = csv.reader(f)
            rows = list(reader)
        with open(RESULTS_CSV, 'w', newline='') as f:
            writer = csv.writer(f)
            for row in rows:
                if row and row[0] != exp_cfg['name']:  # keep header and other experiments
                    writer.writerow(row)

    with open(RESULTS_CSV, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            exp_cfg['name'],
            exp_cfg['description'],
            exp_cfg['alpha'],
            exp_cfg['beta'],
            exp_cfg['gamma'],
            exp_cfg['temperature'],
            os.path.basename(exp_cfg['teacher']) if exp_cfg['teacher'] else 'None',
            round(metrics.get('mAP50', 0), 4),
            round(metrics.get('mAP50_95', 0), 4),
            round(metrics.get('precision', 0), 4),
            round(metrics.get('recall', 0), 4),
            round(metrics.get('fps', 0), 1),
            round(metrics.get('model_size_mb', 0), 1),
            weights_path,
            round(training_time_hrs, 2),
        ])


# ==============================================================================
#  EVALUATION HELPER
# ==============================================================================

def evaluate_checkpoint(weights_path: str, data_yaml: str, device: str) -> dict:
    """Run validation and FPS test on a checkpoint. Returns metrics dict."""
    metrics = {}

    if not os.path.isfile(weights_path):
        print(f"  [WARNING] Checkpoint not found: {weights_path}")
        return metrics

    try:
        model = YOLO(weights_path)

        # Run validation
        val_results = model.val(data=data_yaml, device=device, verbose=False)
        metrics['mAP50']    = float(val_results.box.map50)
        metrics['mAP50_95'] = float(val_results.box.map)
        metrics['precision'] = float(val_results.box.mp)
        metrics['recall']    = float(val_results.box.mr)

        # Measure FPS (batch=1, single image inference)
        use_cuda = device != 'cpu' and torch.cuda.is_available()
        dev = torch.device(f'cuda:{device}') if use_cuda else torch.device('cpu')
        
        if use_cuda:
            model.model.to(dev)

        model.model.eval()
        dtype = next(model.model.parameters()).dtype
        dummy = torch.zeros(1, 3, IMAGE_SIZE, IMAGE_SIZE, device=dev, dtype=dtype)
        
        # Warmup
        with torch.no_grad():
            for _ in range(30):
                _ = model.model(dummy)
        if use_cuda:
            torch.cuda.synchronize(dev)

        # Timed runs
        n_runs = 100
        t0 = time.perf_counter()
        with torch.no_grad():
            for _ in range(n_runs):
                _ = model.model(dummy)
        if use_cuda:
            torch.cuda.synchronize(dev)
        elapsed = time.perf_counter() - t0
        metrics['fps'] = round(n_runs / elapsed, 1)

        # Model size
        metrics['model_size_mb'] = round(os.path.getsize(weights_path) / (1024 * 1024), 1)

    except Exception as e:
        print(f"  [WARNING] Evaluation failed: {e}")

    return metrics


# ==============================================================================
#  RUN ONE EXPERIMENT
# ==============================================================================

def run_experiment(exp_id: str, exp_cfg: dict) -> dict:
    """
    Runs a single experiment. Returns metrics dict.
    """
    print(f"\n{'='*65}")
    print(f"  Running: {exp_id} — {exp_cfg['description']}")
    print(f"{'='*65}")

    # Validate teacher path for KD experiments
    if exp_cfg['mode'] == 'kd':
        teacher_path = exp_cfg['teacher']
        if teacher_path is None or not os.path.isfile(teacher_path):
            print(f"  [SKIP] Teacher checkpoint not found: {teacher_path}")
            print(f"  Set the teacher path in run_ablation.py and re-run.")
            return {}

    run_name = exp_cfg['name']
    weights_path = f"runs/ablation/{run_name}/weights/best.pt"

    t_start = time.time()

    if exp_cfg['mode'] == 'baseline':
        # E1: Standard YOLO training without KD
        model = YOLO("yolov8n.pt")
        model.train(
            data=DATA_YAML,
            epochs=EPOCHS,
            batch=BATCH_SIZE,
            imgsz=IMAGE_SIZE,
            device=DEVICE,
            project="runs/ablation",
            name=run_name,
            exist_ok=True,
            verbose=False,
        )
    else:
        # KD experiments
        trainer = KDTrainer(
            teacher_path=exp_cfg['teacher'],
            alpha=exp_cfg['alpha'],
            beta=exp_cfg['beta'],
            gamma=exp_cfg['gamma'],
            temperature=exp_cfg['temperature'],
            overrides=dict(
                model="yolov8n.pt",
                data=DATA_YAML,
                epochs=EPOCHS,
                batch=BATCH_SIZE,
                imgsz=IMAGE_SIZE,
                device=DEVICE,
                project="runs/ablation",
                name=run_name,
                exist_ok=True,
                verbose=False,
            )
        )
        trainer.train()

    training_time_hrs = (time.time() - t_start) / 3600

    # Evaluate the trained checkpoint
    print(f"\n  Evaluating checkpoint: {weights_path}")
    metrics = evaluate_checkpoint(weights_path, DATA_YAML, DEVICE)

    if metrics:
        print(f"  mAP50    : {metrics.get('mAP50', 'N/A'):.4f}")
        print(f"  mAP50-95 : {metrics.get('mAP50_95', 'N/A'):.4f}")
        print(f"  Precision: {metrics.get('precision', 'N/A'):.4f}")
        print(f"  Recall   : {metrics.get('recall', 'N/A'):.4f}")
        print(f"  FPS      : {metrics.get('fps', 'N/A')}")
        print(f"  Size     : {metrics.get('model_size_mb', 'N/A')} MB")
        print(f"  Time     : {training_time_hrs:.2f} hrs")
    else:
        print("  [WARNING] Could not extract metrics.")

    append_result(exp_id, exp_cfg, metrics, weights_path, training_time_hrs)
    print(f"  Result saved to {RESULTS_CSV}")

    return metrics


# ==============================================================================
#  PRINT RESULTS TABLE
# ==============================================================================

def print_results_table():
    """Reads ablation_results.csv and prints a formatted comparison table."""
    if not os.path.isfile(RESULTS_CSV):
        print("No results found yet.")
        return

    import csv as csv_module

    rows = []
    with open(RESULTS_CSV, 'r') as f:
        reader = csv_module.DictReader(f)
        for row in reader:
            rows.append(row)

    if not rows:
        print("No results in CSV yet.")
        return

    print(f"\n{'='*90}")
    print(f"  ABLATION STUDY RESULTS")
    print(f"{'='*90}")
    print(f"  {'Experiment':<22} {'mAP50':>7} {'mAP50-95':>9} {'Precision':>10} {'Recall':>8} {'FPS':>7} {'Size(MB)':>9}")
    print(f"  {'-'*80}")

    baseline_map = None
    for row in rows:
        name = row.get('experiment', '')[:22]
        map50    = float(row.get('mAP50', 0))
        map5095  = float(row.get('mAP50_95', 0))
        prec     = float(row.get('precision', 0))
        rec      = float(row.get('recall', 0))
        fps      = float(row.get('fps', 0))
        size     = float(row.get('model_size_mb', 0))

        if 'baseline' in name.lower() and baseline_map is None:
            baseline_map = map5095

        delta = ''
        if baseline_map is not None and 'baseline' not in name.lower():
            diff = map5095 - baseline_map
            delta = f"  ({'+' if diff >= 0 else ''}{diff:.4f})"

        print(f"  {name:<22} {map50:>7.4f} {map5095:>9.4f} {prec:>10.4f} {rec:>8.4f} {fps:>7.1f} {size:>9.1f}{delta}")

    print(f"{'='*90}")
    print(f"  delta = change vs E1 baseline (positive = improvement)\n")


# ==============================================================================
#  MAIN
# ==============================================================================

def parse_args():
    parser = argparse.ArgumentParser(description="Run YOLOv8 KD Ablation Study")
    parser.add_argument(
        '--exp', nargs='+', default=None,
        help=(
            "Experiment IDs to run. "
            "Options: E1 E2 E3 E4 E5a E5b E5d E6a E6c E7b E7c. "
            "E5c/E6b/E7a are omitted — they are identical to E4. "
            "Default: E1 E2 E3 E4 E5a E5b E5d E6a E6c"
        )
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Default: run all experiments except E7b and E7c (require extra teacher checkpoints).
    # E5c, E6b, E7a are absent — they are identical to E4, so E4 covers those data points.
    default_exps = ["E1", "E2", "E3", "E4", "E5a", "E5b", "E5d", "E6a", "E6c"]
    experiments_to_run = args.exp if args.exp else default_exps

    # Validate experiment IDs
    unknown = [e for e in experiments_to_run if e not in EXPERIMENTS]
    if unknown:
        print(f"[ERROR] Unknown experiment IDs: {unknown}")
        print(f"  Valid IDs: {list(EXPERIMENTS.keys())}")
        sys.exit(1)

    print("=" * 65)
    print("  YOLOv8 Knowledge Distillation — Ablation Study")
    print("=" * 65)
    print(f"  Experiments to run: {experiments_to_run}")
    print(f"  Epochs per run    : {EPOCHS}")
    print(f"  Results saved to  : {RESULTS_CSV}")
    print("=" * 65)

    # Validate shared config
    if not os.path.isfile(DATA_YAML):
        print(f"\n[ERROR] Dataset YAML not found: {DATA_YAML}")
        sys.exit(1)

    init_results_csv()

    all_results = {}
    for exp_id in experiments_to_run:
        exp_cfg = EXPERIMENTS[exp_id]
        try:
            metrics = run_experiment(exp_id, exp_cfg)
            all_results[exp_id] = metrics
        except KeyboardInterrupt:
            print("\n[INTERRUPTED] Stopped by user. Results saved so far.")
            break
        except Exception as e:
            print(f"\n[ERROR] Experiment {exp_id} failed: {e}")
            print("  Continuing with next experiment...")
            continue

    print_results_table()

    print(f"\nAll done! Full results table: {RESULTS_CSV}")
    print("To view the table again at any time:")
    print("  python -c \"import run_ablation; run_ablation.print_results_table()\"")


if __name__ == "__main__":
    main()