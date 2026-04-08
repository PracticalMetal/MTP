"""
Phase 2 Pipeline: Spectrum-Preserving Slimming for IoT IDS
===========================================================
End-to-end pipeline combining:
  1. Baseline GCN training (Phase 1)
  2. DSR graph rewiring (Technique A — spectral-preserving edge pruning)
  3. SLIDE neuron pruning + de-correlation fine-tuning (Technique B)
  4. Comprehensive evaluation (classification + computational metrics)

Usage:
    python main_pipeline.py [--skip_baseline] [--prune_ratio 0.4] [--device cuda]
"""

import argparse
import os
import time
import json
import copy
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
)

from data_loader import SimulatedIoTDataset, get_dataloaders
from baseline_model import BaselineGCN, train_baseline
from dsr_rewiring import dsr_rewire_dataset
from slide_pruning import (
    create_pruning_masks, apply_pruning_masks, slide_finetune,
    count_active_parameters, compact_model,
)


# ---------------------------------------------------------------------------
# Evaluation utilities
# ---------------------------------------------------------------------------

def evaluate_model(model, test_loader, device="cuda", model_name="Model"):
    """
    Evaluate a model on the test set and return classification metrics.

    Returns
    -------
    metrics : dict
        Accuracy, Precision, Recall, F1, AUC-ROC, confusion matrix.
    """
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            out = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
            probs = F.softmax(out, dim=1)
            preds = out.argmax(dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(batch.y.view(-1).cpu().numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)

    metrics = {
        "accuracy": accuracy_score(all_labels, all_preds),
        "precision": precision_score(all_labels, all_preds, zero_division=0),
        "recall": recall_score(all_labels, all_preds, zero_division=0),
        "f1": f1_score(all_labels, all_preds, zero_division=0),
        "auc_roc": roc_auc_score(all_labels, all_probs),
        "confusion_matrix": confusion_matrix(all_labels, all_preds).tolist(),
    }

    print(f"\n{'='*60}")
    print(f"  {model_name} — Classification Metrics")
    print(f"{'='*60}")
    print(f"  Accuracy:  {metrics['accuracy']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall:    {metrics['recall']:.4f}")
    print(f"  F1-Score:  {metrics['f1']:.4f}")
    print(f"  AUC-ROC:   {metrics['auc_roc']:.4f}")
    print(f"  Confusion Matrix:")
    cm = metrics["confusion_matrix"]
    print(f"    TN={cm[0][0]:5d}  FP={cm[0][1]:5d}")
    print(f"    FN={cm[1][0]:5d}  TP={cm[1][1]:5d}")
    print(f"{'='*60}")

    return metrics


def measure_computational_metrics(model, sample_batch, device="cuda",
                                  num_warmup=10, num_runs=100,
                                  masks=None):
    """
    Measure computational metrics: parameter count, memory, inference time.

    Parameters
    ----------
    model : nn.Module
    sample_batch : PyG Batch
        A single batch for inference timing.
    device : str
    num_warmup, num_runs : int
        Warmup and measurement iterations for timing.
    masks : dict or None
        If provided, count only active (non-pruned) parameters.

    Returns
    -------
    comp_metrics : dict
        param_count, active_param_count, memory_mb, inference_ms
    """
    model.eval()
    sample_batch = sample_batch.to(device)

    # Parameter count
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    active_params = count_active_parameters(model, masks) if masks else total_params

    # Memory footprint (model size in MB)
    param_size = sum(p.nelement() * p.element_size() for p in model.parameters())
    buffer_size = sum(b.nelement() * b.element_size() for b in model.buffers())
    memory_mb = (param_size + buffer_size) / (1024 ** 2)

    # Inference time
    model.to(device)
    with torch.no_grad():
        # Warmup
        for _ in range(num_warmup):
            _ = model(sample_batch.x, sample_batch.edge_index,
                      sample_batch.edge_attr, sample_batch.batch)

        if device == "cuda":
            torch.cuda.synchronize()

        times = []
        for _ in range(num_runs):
            if device == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            _ = model(sample_batch.x, sample_batch.edge_index,
                      sample_batch.edge_attr, sample_batch.batch)
            if device == "cuda":
                torch.cuda.synchronize()
            t1 = time.perf_counter()
            times.append((t1 - t0) * 1000)  # ms

    inference_ms = np.mean(times)
    inference_std = np.std(times)

    comp_metrics = {
        "total_params": total_params,
        "active_params": active_params,
        "memory_mb": round(memory_mb, 4),
        "inference_ms": round(inference_ms, 4),
        "inference_std_ms": round(inference_std, 4),
    }

    print(f"  Parameters (total):  {total_params:,}")
    print(f"  Parameters (active): {active_params:,}")
    print(f"  Memory footprint:    {memory_mb:.4f} MB")
    print(f"  Inference time:      {inference_ms:.4f} ± {inference_std:.4f} ms")

    return comp_metrics


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def main(args):
    device = args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu"
    print(f"[Pipeline] Device: {device}")
    print(f"[Pipeline] Prune ratio: {args.prune_ratio}")
    print(f"[Pipeline] Lambda decorr: {args.lambda_decorr}")

    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs("results", exist_ok=True)

    # ------------------------------------------------------------------
    # Step 0: Load dataset
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("  STEP 0: Loading Dataset")
    print("=" * 60)

    dataset = SimulatedIoTDataset(
        root=args.data_root,
        num_graphs=args.num_graphs,
        min_nodes=args.min_nodes,
        max_nodes=args.max_nodes,
        seed=args.seed,
    )
    in_channels = dataset[0].x.size(1)

    train_loader, val_loader, test_loader = get_dataloaders(
        dataset, batch_size=args.batch_size, seed=args.seed,
    )

    # Get a sample batch for inference timing
    sample_batch = next(iter(test_loader))

    # ------------------------------------------------------------------
    # Step 1: Train or load baseline GCN
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("  STEP 1: Baseline GCN (Phase 1)")
    print("=" * 60)

    baseline = BaselineGCN(
        in_channels=in_channels,
        hidden_channels=args.hidden_channels,
        num_classes=2,
        num_gcn_layers=args.num_gcn_layers,
        dropout=args.dropout,
    )

    baseline_path = "checkpoints/baseline_gcn.pt"
    if args.skip_baseline and os.path.exists(baseline_path):
        print(f"[Baseline] Loading from {baseline_path}")
        baseline.load_state_dict(
            torch.load(baseline_path, map_location=device, weights_only=True)
        )
        baseline = baseline.to(device)
    else:
        print("[Baseline] Training from scratch...")
        baseline, baseline_history = train_baseline(
            baseline, train_loader, val_loader,
            epochs=args.baseline_epochs, lr=args.lr, device=device,
            save_path=baseline_path,
        )

    # Evaluate baseline
    baseline_metrics = evaluate_model(baseline, test_loader, device, "Baseline GCN")
    print("\n  Baseline Computational Metrics:")
    baseline_comp = measure_computational_metrics(
        baseline, sample_batch, device
    )

    # ------------------------------------------------------------------
    # Step 2: DSR Graph Rewiring (Technique A)
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("  STEP 2: DSR Graph Rewiring")
    print("=" * 60)

    print("[DSR] Rewiring training graphs...")
    train_data_list = [train_loader.dataset[i] for i in range(len(train_loader.dataset))]
    val_data_list = [val_loader.dataset[i] for i in range(len(val_loader.dataset))]
    test_data_list = [test_loader.dataset[i] for i in range(len(test_loader.dataset))]

    dsr_kwargs = {
        "shortcut_ratio": args.shortcut_ratio,
        "k_eigen": args.k_eigen,
        "epsilon": args.epsilon,
        "seed": args.seed,
    }

    train_rewired = dsr_rewire_dataset(train_data_list, **dsr_kwargs)
    val_rewired = dsr_rewire_dataset(val_data_list, **dsr_kwargs)
    test_rewired = dsr_rewire_dataset(test_data_list, **dsr_kwargs)

    from torch_geometric.loader import DataLoader
    train_loader_dsr = DataLoader(train_rewired, batch_size=args.batch_size, shuffle=True)
    val_loader_dsr = DataLoader(val_rewired, batch_size=args.batch_size, shuffle=False)
    test_loader_dsr = DataLoader(test_rewired, batch_size=args.batch_size, shuffle=False)

    # Re-train baseline on rewired graphs for fair comparison
    print("\n[DSR] Training baseline on rewired graphs...")
    baseline_dsr = BaselineGCN(
        in_channels=in_channels,
        hidden_channels=args.hidden_channels,
        num_classes=2,
        num_gcn_layers=args.num_gcn_layers,
        dropout=args.dropout,
    )
    baseline_dsr, dsr_history = train_baseline(
        baseline_dsr, train_loader_dsr, val_loader_dsr,
        epochs=args.baseline_epochs, lr=args.lr, device=device,
        save_path="checkpoints/baseline_dsr.pt",
    )

    dsr_metrics = evaluate_model(baseline_dsr, test_loader_dsr, device, "Baseline + DSR")
    print("\n  DSR Computational Metrics:")
    sample_batch_dsr = next(iter(test_loader_dsr))
    dsr_comp = measure_computational_metrics(
        baseline_dsr, sample_batch_dsr, device
    )

    # ------------------------------------------------------------------
    # Step 3: SLIDE Neuron Pruning + Fine-tuning (Technique B)
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("  STEP 3: SLIDE Pruning + De-correlation Fine-tuning")
    print("=" * 60)

    # Step 3a: Identify neurons to prune from the DSR-trained model
    model_to_prune = copy.deepcopy(baseline_dsr)

    print(f"[SLIDE] Creating pruning masks (ratio={args.prune_ratio}, "
          f"strategy={args.prune_strategy})...")
    masks = create_pruning_masks(
        model_to_prune,
        prune_ratio=args.prune_ratio,
        strategy=args.prune_strategy,
        seed=args.seed,
    )

    # Report pruning summary
    for name, mask in masks.items():
        kept = int(mask.sum().item())
        total = mask.numel()
        print(f"  Layer '{name}': {kept}/{total} neurons kept "
              f"({100*kept/total:.1f}%)")

    # Step 3b: Compact — physically remove pruned neurons
    print("\n[SLIDE] Compacting model — physically removing pruned neurons...")
    model_compact = compact_model(model_to_prune, masks)
    model_compact = model_compact.to(device)

    # Step 3c: Fine-tune the compact model with de-correlation loss
    print("\n[SLIDE] Fine-tuning compact model with de-correlation loss...")
    model_slide, slide_history = slide_finetune(
        model_compact, train_loader_dsr, val_loader_dsr,
        masks=None,  # No masks needed — model is already physically compact
        epochs=args.slide_epochs,
        lr=args.slide_lr,
        lambda_decorr=args.lambda_decorr,
        rff_dim=args.rff_dim,
        patience=args.slide_patience,
        device=device,
        save_path="checkpoints/slide_compact.pt",
    )

    # Evaluate compact fine-tuned model
    compact_metrics = evaluate_model(model_slide, test_loader_dsr, device, "DSR + SLIDE (compact)")
    print("\n  Compact Model Computational Metrics:")
    compact_comp = measure_computational_metrics(
        model_slide, sample_batch_dsr, device
    )

    # ------------------------------------------------------------------
    # Step 4: Summary comparison
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("  FINAL COMPARISON")
    print("=" * 60)

    header = f"{'Metric':<25} {'Baseline':>12} {'+ DSR':>12} {'+ DSR+SLIDE':>14}"
    print(header)
    print("-" * 65)

    for metric_name in ["accuracy", "precision", "recall", "f1", "auc_roc"]:
        v1 = baseline_metrics[metric_name]
        v2 = dsr_metrics[metric_name]
        v3 = compact_metrics[metric_name]
        print(f"  {metric_name:<23} {v1:>12.4f} {v2:>12.4f} {v3:>14.4f}")

    print("-" * 65)
    print(f"  {'Total Params':<23} {baseline_comp['total_params']:>12,} "
          f"{dsr_comp['total_params']:>12,} {compact_comp['total_params']:>14,}")
    print(f"  {'Memory (MB)':<23} {baseline_comp['memory_mb']:>12.4f} "
          f"{dsr_comp['memory_mb']:>12.4f} {compact_comp['memory_mb']:>14.4f}")
    print(f"  {'Inference (ms)':<23} {baseline_comp['inference_ms']:>12.4f} "
          f"{dsr_comp['inference_ms']:>12.4f} {compact_comp['inference_ms']:>14.4f}")

    # Compression ratio
    compression = baseline_comp['total_params'] / max(1, compact_comp['total_params'])
    mem_compression = baseline_comp['memory_mb'] / max(0.0001, compact_comp['memory_mb'])
    speedup = baseline_comp['inference_ms'] / max(0.001, compact_comp['inference_ms'])
    print(f"\n  Parameter compression: {compression:.2f}x")
    print(f"  Memory compression:    {mem_compression:.2f}x")
    print(f"  Speedup:               {speedup:.2f}x")

    # ------------------------------------------------------------------
    # Save results
    # ------------------------------------------------------------------
    results = {
        "baseline": {**baseline_metrics, **baseline_comp},
        "dsr": {**dsr_metrics, **dsr_comp},
        "slide_compact": {**compact_metrics, **compact_comp},
        "param_compression": compression,
        "memory_compression": mem_compression,
        "speedup": speedup,
        "config": vars(args),
    }

    results_path = "results/phase2_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n[Pipeline] Results saved to {results_path}")

    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Phase 2: Spectrum-Preserving Slimming for IoT IDS"
    )

    # Data
    parser.add_argument("--data_root", type=str, default="./data/simulated")
    parser.add_argument("--num_graphs", type=int, default=2000)
    parser.add_argument("--min_nodes", type=int, default=20)
    parser.add_argument("--max_nodes", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=64)

    # Model
    parser.add_argument("--hidden_channels", type=int, default=128)
    parser.add_argument("--num_gcn_layers", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.5)

    # Baseline training
    parser.add_argument("--skip_baseline", action="store_true",
                        help="Skip baseline training if checkpoint exists")
    parser.add_argument("--baseline_epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-3)

    # DSR
    parser.add_argument("--shortcut_ratio", type=float, default=0.1)
    parser.add_argument("--k_eigen", type=int, default=50)
    parser.add_argument("--epsilon", type=float, default=0.3)

    # SLIDE
    parser.add_argument("--prune_ratio", type=float, default=0.4)
    parser.add_argument("--prune_strategy", type=str, default="magnitude_random",
                        choices=["random", "magnitude", "magnitude_random"])
    parser.add_argument("--slide_epochs", type=int, default=100)
    parser.add_argument("--slide_lr", type=float, default=1e-3)
    parser.add_argument("--lambda_decorr", type=float, default=0.01)
    parser.add_argument("--rff_dim", type=int, default=256)
    parser.add_argument("--slide_patience", type=int, default=15)

    # General
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # Reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    main(args)
