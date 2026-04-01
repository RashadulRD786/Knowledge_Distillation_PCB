"""
evaluate.py — Model Evaluation Script
======================================
Evaluate any YOLOv8 checkpoint on the PCB test set and display all key metrics.

HOW TO RUN:
-----------
    # Evaluate the KD-trained student:
    python evaluate.py --weights runs/kd/kd_training/weights/best.pt

    # Compare baseline vs KD student side by side:
    python evaluate.py --weights runs/ablation/E1_baseline/weights/best.pt
                                 runs/ablation/E4_combined_kd/weights/best.pt

    # Evaluate on val split instead of test:
    python evaluate.py --weights runs/kd/kd_training/weights/best.pt --split val

OUTPUT:
-------
    Prints a detailed metrics table to the terminal.
    Optionally saves results to a JSON file.
"""

import os
import sys
import time
import json
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
from ultralytics import YOLO


# ==============================================================================
#  CONFIGURATION
# ==============================================================================

DATA_YAML  = "data/pcb_defect.yaml"
IMAGE_SIZE = 640
DEVICE     = "0"


# ==============================================================================
#  EVALUATION FUNCTION
# ==============================================================================

def evaluate_model(weights_path: str, data_yaml: str, split: str = "test",
                   device: str = "0", imgsz: int = 640) -> dict:
    """
    Evaluate a YOLOv8 checkpoint and return a dict of metrics.

    Args:
        weights_path : path to the .pt checkpoint file
        data_yaml    : path to the dataset YAML config
        split        : which split to evaluate on ("test" or "val")
        device       : GPU device string ("0", "1", "cpu")
        imgsz        : input image size

    Returns:
        dict with keys: mAP50, mAP50_95, precision, recall, fps, model_size_mb,
                        per_class_ap (dict), inference_ms
    """
    if not os.path.isfile(weights_path):
        print(f"[ERROR] Checkpoint not found: {weights_path}")
        return {}

    print(f"\nLoading model: {weights_path}")
    model = YOLO(weights_path)
    model_size_mb = round(os.path.getsize(weights_path) / (1024 * 1024), 2)

    # ── Validation / Test metrics ──────────────────────────────────────────────
    print(f"Running validation on {split} split...")
    val_results = model.val(
        data=data_yaml,
        split=split,
        imgsz=imgsz,
        device=device,
        verbose=True,
    )

    mAP50    = float(val_results.box.map50)
    mAP5095  = float(val_results.box.map)
    precision = float(val_results.box.mp)
    recall    = float(val_results.box.mr)

    # Per-class AP at IoU=0.5
    per_class_ap = {}
    class_names = val_results.names if hasattr(val_results, 'names') else {}
    if hasattr(val_results.box, 'ap50'):
        ap50_per_class = val_results.box.ap50
        class_indices = getattr(val_results.box, 'ap_class_index', range(len(ap50_per_class)))
        for c_idx, ap in zip(class_indices, ap50_per_class):
            cls_name = class_names.get(int(c_idx), f"class_{c_idx}") if class_names else f"class_{c_idx}"
            per_class_ap[cls_name] = round(float(ap), 4)

    # ── FPS measurement ────────────────────────────────────────────────────────
    print("Measuring inference speed (FPS)...")
    model.model.eval()
    use_cuda = device != 'cpu' and torch.cuda.is_available()
    dev = torch.device(f'cuda:{device}') if use_cuda else torch.device('cpu')
    dtype = next(model.model.parameters()).dtype
    dummy = torch.zeros(1, 3, imgsz, imgsz, device=dev, dtype=dtype)
    if use_cuda:
        model.model.to(dev)

    # Warmup runs
    with torch.no_grad():
        for _ in range(50):
            _ = model.model(dummy)
    if use_cuda:
        torch.cuda.synchronize(dev)

    # Timed runs
    n_runs = 200
    t0 = time.perf_counter()
    with torch.no_grad():
        for _ in range(n_runs):
            _ = model.model(dummy)
    if use_cuda:
        torch.cuda.synchronize(dev)
    elapsed = time.perf_counter() - t0

    fps = round(n_runs / elapsed, 1)
    inference_ms = round((elapsed / n_runs) * 1000, 2)

    results = {
        "weights_path"  : weights_path,
        "model_size_mb" : model_size_mb,
        "mAP50"         : round(mAP50, 4),
        "mAP50_95"      : round(mAP5095, 4),
        "precision"     : round(precision, 4),
        "recall"        : round(recall, 4),
        "fps"           : fps,
        "inference_ms"  : inference_ms,
        "per_class_ap"  : per_class_ap,
        "split"         : split,
    }

    return results


# ==============================================================================
#  DISPLAY FUNCTIONS
# ==============================================================================

def print_single_result(results: dict):
    """Print a formatted single-model results block."""
    if not results:
        return

    print("\n" + "=" * 60)
    print(f"  EVALUATION RESULTS")
    print("=" * 60)
    print(f"  Model        : {os.path.basename(results['weights_path'])}")
    print(f"  Split        : {results['split']}")
    print(f"  Model size   : {results['model_size_mb']} MB")
    print("-" * 60)
    print(f"  mAP50        : {results['mAP50']:.4f}")
    print(f"  mAP50-95     : {results['mAP50_95']:.4f}")
    print(f"  Precision    : {results['precision']:.4f}")
    print(f"  Recall       : {results['recall']:.4f}")
    print("-" * 60)
    print(f"  FPS          : {results['fps']}  (forward pass only, excludes NMS)")
    print(f"  Inference    : {results['inference_ms']} ms/image")
    print("-" * 60)

    if results['per_class_ap']:
        print("  Per-class AP@0.5:")
        for cls_name, ap in results['per_class_ap'].items():
            bar = "█" * int(ap * 30)
            print(f"    {cls_name:<20} {ap:.4f}  {bar}")

    print("=" * 60)


def print_comparison_table(all_results: list):
    """Print a side-by-side comparison table for multiple models."""
    if len(all_results) < 2:
        return

    print("\n" + "=" * 90)
    print("  MODEL COMPARISON")
    print("=" * 90)
    header = f"  {'Model':<35} {'mAP50':>7} {'mAP50-95':>9} {'Precision':>10} {'Recall':>8} {'FPS':>7} {'Size':>7}"
    print(header)
    print("  " + "-" * 85)

    baseline = all_results[0]
    for i, res in enumerate(all_results):
        name = os.path.basename(res['weights_path'])[:35]
        map50   = res['mAP50']
        map5095 = res['mAP50_95']
        prec    = res['precision']
        rec     = res['recall']
        fps     = res['fps']
        size    = res['model_size_mb']

        delta = ''
        if i > 0:
            diff = map5095 - baseline['mAP50_95']
            sign = '+' if diff >= 0 else ''
            delta = f"  ({sign}{diff:.4f})"

        print(f"  {name:<35} {map50:>7.4f} {map5095:>9.4f} {prec:>10.4f} {rec:>8.4f} {fps:>7.1f} {size:>6.1f}MB{delta}")

    print("=" * 90)
    print("  delta = mAP50-95 change vs first model listed\n")


# ==============================================================================
#  MAIN
# ==============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate YOLOv8 checkpoints on PCB defect dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python evaluate.py --weights runs/kd/kd_training/weights/best.pt
  python evaluate.py --weights model_a.pt model_b.pt --split val
  python evaluate.py --weights best.pt --save results.json
        """
    )
    parser.add_argument(
        '--weights', nargs='+', required=True,
        help="Path(s) to .pt checkpoint file(s). Multiple paths = comparison mode."
    )
    parser.add_argument(
        '--data', default=DATA_YAML,
        help=f"Path to dataset YAML (default: {DATA_YAML})"
    )
    parser.add_argument(
        '--split', default='test', choices=['train', 'val', 'test'],
        help="Dataset split to evaluate on (default: test)"
    )
    parser.add_argument(
        '--device', default=DEVICE,
        help="Device to use: '0' for GPU 0, 'cpu' for CPU (default: 0)"
    )
    parser.add_argument(
        '--imgsz', type=int, default=IMAGE_SIZE,
        help=f"Input image size (default: {IMAGE_SIZE})"
    )
    parser.add_argument(
        '--save', default=None,
        help="Optional: save results to a JSON file (e.g., --save results.json)"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if not os.path.isfile(args.data):
        print(f"[ERROR] Dataset YAML not found: {args.data}")
        sys.exit(1)

    all_results = []

    for weights_path in args.weights:
        results = evaluate_model(
            weights_path=weights_path,
            data_yaml=args.data,
            split=args.split,
            device=args.device,
            imgsz=args.imgsz,
        )
        if results:
            all_results.append(results)
            if len(args.weights) == 1:
                print_single_result(results)

    if len(all_results) > 1:
        for r in all_results:
            print_single_result(r)
        print_comparison_table(all_results)

    if args.save and all_results:
        with open(args.save, 'w') as f:
            json.dump(all_results, f, indent=2)
        print(f"Results saved to: {args.save}")

    if not all_results:
        print("\n[ERROR] No models could be evaluated. Check your --weights paths.")
        sys.exit(1)


if __name__ == "__main__":
    main()