"""
Reconstruction-Aware Differentiable Pruning GNN — Main Pipeline
================================================================
End-to-end pipeline:
  0. Generate synthetic IoT graph with node-level labels
  1. MLP baseline (no graph structure) — validates GNN adds value
  2. Dense GCN baseline (all edges, no pruning)
  3. PruningGNN with temperature annealing + tri-partite loss
  4. Comprehensive evaluation + noise edge analysis

Usage:
    python main_pipeline.py [--num_nodes 5000] [--epochs 300] [--device cuda]
"""

import argparse
import os
import time
import json
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix,
)

from node_level_data import generate_iot_graph
from reconstruction_pruning_model import (
    PruningGNN, GNNBackbone, tripartite_loss,
)


# ---------------------------------------------------------------------------
# MLP Baseline (no graph structure)
# ---------------------------------------------------------------------------
class MLPBaseline(nn.Module):
    """Simple 3-layer MLP on node features only. No graph structure."""
    def __init__(self, in_channels, hidden=128, num_classes=2, dropout=0.5):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_channels, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden // 2, num_classes),
        )

    def forward(self, x):
        return self.net(x)


# ---------------------------------------------------------------------------
# Temperature annealing
# ---------------------------------------------------------------------------
def get_temperature(epoch, total_epochs, tau_start=1.0, tau_end=0.1):
    decay = (tau_end / tau_start) ** (epoch / max(1, total_epochs - 1))
    return tau_start * decay


# ---------------------------------------------------------------------------
# Train MLP baseline
# ---------------------------------------------------------------------------
def train_mlp(model, data, epochs=200, lr=1e-3, weight_decay=5e-4,
              patience=50, device="cuda"):
    model = model.to(device)
    data = data.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_val_f1 = 0.0
    best_state = None
    wait = 0

    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()
        logits = model(data.x)
        loss = F.cross_entropy(logits[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()

        if epoch % 5 == 0 or epoch == 1:
            model.eval()
            with torch.no_grad():
                val_logits = model(data.x)
                val_preds = val_logits[data.val_mask].argmax(dim=1).cpu().numpy()
                val_y = data.y[data.val_mask].cpu().numpy()
                val_f1 = f1_score(val_y, val_preds, zero_division=0)
            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                best_state = copy.deepcopy(model.state_dict())
                wait = 0
            else:
                wait += 5
                if wait >= patience:
                    break
        if epoch % 50 == 0:
            print(f"  [MLP] Epoch {epoch}/{epochs} | Loss: {loss.item():.4f} | Val F1: {val_f1:.4f}")

    if best_state:
        model.load_state_dict(best_state)
    print(f"  [MLP] Best val F1: {best_val_f1:.4f}")
    return model


# ---------------------------------------------------------------------------
# Train Dense GCN baseline
# ---------------------------------------------------------------------------
def train_dense_baseline(model, data, epochs=200, lr=1e-3, weight_decay=5e-4,
                         patience=50, device="cuda"):
    model = model.to(device)
    data = data.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_val_f1 = 0.0
    best_state = None
    wait = 0

    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()
        logits = model(data.x, data.edge_index, edge_weight=data.edge_attr)
        loss = F.cross_entropy(logits[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()

        if epoch % 5 == 0 or epoch == 1:
            model.eval()
            with torch.no_grad():
                val_logits = model(data.x, data.edge_index, edge_weight=data.edge_attr)
                val_preds = val_logits[data.val_mask].argmax(dim=1).cpu().numpy()
                val_y = data.y[data.val_mask].cpu().numpy()
                val_f1 = f1_score(val_y, val_preds, zero_division=0)
            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                best_state = copy.deepcopy(model.state_dict())
                wait = 0
            else:
                wait += 5
                if wait >= patience:
                    print(f"  [Dense] Early stop at epoch {epoch} (best val F1: {best_val_f1:.4f})")
                    break
        if epoch % 50 == 0:
            print(f"  [Dense] Epoch {epoch}/{epochs} | Loss: {loss.item():.4f} | Val F1: {val_f1:.4f}")

    if best_state:
        model.load_state_dict(best_state)
    return model


# ---------------------------------------------------------------------------
# Train Pruning GNN
# ---------------------------------------------------------------------------
def train_pruning_gnn(model, data, epochs=200, lr=1e-3, weight_decay=5e-4,
                      lambda1=0.1, lambda2=0.1, target_sparsity=0.3,
                      tau_start=1.0, tau_end=0.1,
                      patience=80, min_epochs=100,
                      warmup_epochs=50, class_conditional_recon=True,
                      normal_class=0, device="cuda"):
    """
    2-phase training for PruningGNN with scout GCN:

    Phase 1 (warmup, epochs 1..warmup_epochs):
        Train all components without sparsity loss (lambda2=0).
        The scout GCN learns to classify nodes, AE learns reconstructions.

    Phase 2 (pruning, remaining epochs):
        Enable sparsity loss, anneal temperature.
        Scout GCN's prediction agreement guides selective edge pruning.
    """
    model = model.to(device)
    data = data.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_val_f1 = 0.0
    best_state = None
    wait = 0
    history = {"loss": [], "l_ce": [], "l_recon": [], "l_sparse": [],
               "sparsity": [], "tau": [], "val_f1": []}

    for epoch in range(1, epochs + 1):
        in_warmup = epoch <= warmup_epochs
        if in_warmup:
            phase = "WARMUP"
            tau = 1.0
            eff_lambda2 = 0.0
        else:
            phase = "PRUNE"
            prune_epoch = epoch - warmup_epochs
            prune_total = epochs - warmup_epochs
            tau = get_temperature(prune_epoch, prune_total, tau_start, tau_end)
            eff_lambda2 = lambda2

        model.train()
        optimizer.zero_grad()

        out = model(data.x, data.edge_index, data.edge_attr, tau=tau)

        total_loss, l_ce, l_recon, l_sparse = tripartite_loss(
            logits=out["logits"], y=data.y, mask=data.train_mask,
            x=data.x, x_recon=out["x_recon"],
            keep_probs=out["keep_probs"], edge_index=data.edge_index,
            lambda1=lambda1, lambda2=eff_lambda2,
            target_sparsity=target_sparsity,
            normal_class=normal_class,
            class_conditional_recon=class_conditional_recon,
        )

        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        history["loss"].append(total_loss.item())
        history["l_ce"].append(l_ce.item())
        history["l_recon"].append(l_recon.item())
        history["l_sparse"].append(l_sparse.item() if torch.is_tensor(l_sparse) else l_sparse)
        history["sparsity"].append(out["sparsity"])
        history["tau"].append(tau)

        if epoch % 5 == 0 or epoch == 1:
            model.eval()
            with torch.no_grad():
                val_out = model(data.x, data.edge_index, data.edge_attr, tau=tau)
                val_preds = val_out["logits"][data.val_mask].argmax(dim=1).cpu().numpy()
                val_y = data.y[data.val_mask].cpu().numpy()
                val_f1 = f1_score(val_y, val_preds, zero_division=0)
                history["val_f1"].append(val_f1)

            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                best_state = copy.deepcopy(model.state_dict())
                wait = 0
            else:
                wait += 5
                if wait >= patience and not in_warmup and epoch >= min_epochs:
                    print(f"  [Pruning] Early stop at epoch {epoch} (best val F1: {best_val_f1:.4f})")
                    break

        if epoch % 25 == 0:
            print(f"  [{phase}] Epoch {epoch}/{epochs} | "
                  f"Loss: {total_loss.item():.4f} (CE: {l_ce.item():.4f}, "
                  f"Recon: {l_recon.item():.4f}, Sparse: {l_sparse.item() if torch.is_tensor(l_sparse) else l_sparse:.4f}) | "
                  f"Sparsity: {out['sparsity']:.2%} | tau: {tau:.4f} | "
                  f"Val F1: {val_f1:.4f}")

    if best_state:
        model.load_state_dict(best_state)
    return model, history


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
def evaluate_node_classification(model, data, mask, device="cuda",
                                 model_name="Model", is_pruning=False,
                                 is_mlp=False, tau=0.1):
    """Evaluate node classification metrics."""
    model.eval()
    data = data.to(device)

    with torch.no_grad():
        if is_mlp:
            logits = model(data.x)
            sparsity = 0.0
        elif is_pruning:
            out = model(data.x, data.edge_index, data.edge_attr, tau=tau)
            logits = out["logits"]
            sparsity = out["sparsity"]
        else:
            logits = model(data.x, data.edge_index, edge_weight=data.edge_attr)
            sparsity = 0.0

    preds = logits[mask].argmax(dim=1).cpu().numpy()
    probs = F.softmax(logits[mask], dim=1)[:, 1].cpu().numpy()
    y_true = data.y[mask].cpu().numpy()

    metrics = {
        "accuracy": accuracy_score(y_true, preds),
        "precision": precision_score(y_true, preds, zero_division=0),
        "recall": recall_score(y_true, preds, zero_division=0),
        "f1": f1_score(y_true, preds, zero_division=0),
        "auc_roc": roc_auc_score(y_true, probs) if len(np.unique(y_true)) > 1 else 0.0,
        "confusion_matrix": confusion_matrix(y_true, preds).tolist(),
        "sparsity": sparsity,
    }

    print(f"\n{'='*60}")
    print(f"  {model_name} — Node Classification Metrics")
    print(f"{'='*60}")
    print(f"  Accuracy:  {metrics['accuracy']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall:    {metrics['recall']:.4f}")
    print(f"  F1-Score:  {metrics['f1']:.4f}")
    print(f"  AUC-ROC:   {metrics['auc_roc']:.4f}")
    if is_pruning:
        print(f"  Sparsity:  {metrics['sparsity']:.2%} edges pruned")
    cm = metrics["confusion_matrix"]
    print(f"  Confusion Matrix:")
    print(f"    TN={cm[0][0]:5d}  FP={cm[0][1]:5d}")
    print(f"    FN={cm[1][0]:5d}  TP={cm[1][1]:5d}")
    print(f"{'='*60}")
    return metrics


def measure_inference(model, data, device="cuda", num_warmup=20, num_runs=100,
                      is_pruning=False, is_mlp=False, tau=0.1, model_name="Model"):
    """Measure inference time and memory."""
    model.eval()
    data = data.to(device)

    total_params = sum(p.numel() for p in model.parameters())
    param_size = sum(p.nelement() * p.element_size() for p in model.parameters())
    buffer_size = sum(b.nelement() * b.element_size() for b in model.buffers())
    memory_mb = (param_size + buffer_size) / (1024 ** 2)

    def _run():
        if is_mlp:
            return model(data.x)
        elif is_pruning:
            return model(data.x, data.edge_index, data.edge_attr, tau=tau)
        else:
            return model(data.x, data.edge_index, edge_weight=data.edge_attr)

    with torch.no_grad():
        for _ in range(num_warmup):
            _run()
        if device == "cuda":
            torch.cuda.synchronize()

        times = []
        for _ in range(num_runs):
            if device == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            out = _run()
            if device == "cuda":
                torch.cuda.synchronize()
            times.append((time.perf_counter() - t0) * 1000)

    # Count edges at eval
    if is_pruning:
        with torch.no_grad():
            out = model(data.x, data.edge_index, data.edge_attr, tau=tau)
            kept_edges = out["pruned_edge_index"].size(1)
            total_edges = data.edge_index.size(1)
    else:
        kept_edges = data.edge_index.size(1)
        total_edges = kept_edges

    inf_ms = np.mean(times)
    inf_std = np.std(times)

    comp = {
        "total_params": total_params,
        "memory_mb": round(memory_mb, 4),
        "inference_ms": round(inf_ms, 4),
        "inference_std_ms": round(inf_std, 4),
        "kept_edges": kept_edges,
        "total_edges": total_edges,
    }

    print(f"\n  {model_name} — Performance:")
    print(f"  Parameters:     {total_params:,}")
    print(f"  Memory:         {memory_mb:.4f} MB")
    print(f"  Inference:      {inf_ms:.4f} +/- {inf_std:.4f} ms")
    if is_pruning:
        print(f"  Edges kept:     {kept_edges:,} / {total_edges:,} "
              f"({100*kept_edges/max(1,total_edges):.1f}%)")
    return comp


def analyze_noise_pruning(model, data, device="cuda", tau=0.1):
    """Check if pruning preferentially removes noisy cross-class edges."""
    model.eval()
    data = data.to(device)

    with torch.no_grad():
        out = model(data.x, data.edge_index, data.edge_attr, tau=tau)
        edge_mask = out["edge_mask"]

    is_noisy = data.is_noisy_edge.to(device)
    is_self_loop = (data.edge_index[0] == data.edge_index[1]).to(device)

    # Only analyze non-self-loop edges
    regular = ~is_self_loop
    noisy_and_regular = is_noisy & regular
    clean_and_regular = (~is_noisy) & regular

    noisy_kept = edge_mask[noisy_and_regular].mean().item() if noisy_and_regular.sum() > 0 else 0
    clean_kept = edge_mask[clean_and_regular].mean().item() if clean_and_regular.sum() > 0 else 0
    noisy_pruned_pct = 1.0 - noisy_kept
    clean_pruned_pct = 1.0 - clean_kept

    print(f"\n  Noise Edge Analysis:")
    print(f"  Noisy cross-class edges pruned: {noisy_pruned_pct:.2%}")
    print(f"  Clean same-class edges pruned:  {clean_pruned_pct:.2%}")
    print(f"  Selectivity ratio:              {noisy_pruned_pct / max(0.001, clean_pruned_pct):.2f}x")
    return {"noisy_pruned": noisy_pruned_pct, "clean_pruned": clean_pruned_pct}


# ---------------------------------------------------------------------------
# Main Pipeline
# ---------------------------------------------------------------------------
def main(args):
    device = args.device if torch.cuda.is_available() else "cpu"
    print(f"[Pipeline] Device: {device}")
    print(f"[Pipeline] Nodes: {args.num_nodes} | Attack ratio: {args.attack_ratio}")
    print(f"[Pipeline] Noise edge ratio: {args.noise_edge_ratio}")
    print(f"[Pipeline] Lambda1 (recon): {args.lambda1} | Lambda2 (sparse): {args.lambda2}")
    print(f"[Pipeline] Tau: {args.tau_start} -> {args.tau_end}")
    print(f"[Pipeline] Class-conditional AE recon: {args.class_conditional_recon} "
          f"(normal-class only = self-supervised pruning signal)")

    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs("results", exist_ok=True)

    # ------------------------------------------------------------------
    # Step 0: Generate data
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("  STEP 0: Generating IoT Network Graph")
    print("=" * 60)

    data = generate_iot_graph(
        num_nodes=args.num_nodes,
        edge_density=args.edge_density,
        attack_ratio=args.attack_ratio,
        feature_dim=args.feature_dim,
        noise_edge_ratio=args.noise_edge_ratio,
        seed=args.seed,
    )
    in_channels = data.x.size(1)
    print(f"  Feature dim: {in_channels}")

    # ------------------------------------------------------------------
    # Step 1: MLP Baseline (features only, no graph)
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("  STEP 1: MLP Baseline (No Graph Structure)")
    print("=" * 60)

    mlp = MLPBaseline(in_channels, hidden=128, num_classes=2, dropout=0.5)
    mlp = train_mlp(mlp, data, epochs=args.epochs, lr=args.lr,
                     patience=args.patience, device=device)
    mlp_metrics = evaluate_node_classification(
        mlp, data, data.test_mask, device, "MLP (features only)", is_mlp=True)
    mlp_comp = measure_inference(mlp, data, device, is_mlp=True, model_name="MLP")

    # ------------------------------------------------------------------
    # Step 2: Dense GCN Baseline
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("  STEP 2: Dense GCN Baseline (All Edges)")
    print("=" * 60)

    dense_model = GNNBackbone(in_channels, args.gnn_hidden, 2, args.dropout)
    print(f"  Training for {args.epochs} epochs...")
    dense_model = train_dense_baseline(
        dense_model, data, epochs=args.epochs, lr=args.lr,
        weight_decay=args.weight_decay, patience=args.patience, device=device)
    dense_metrics = evaluate_node_classification(
        dense_model, data, data.test_mask, device, "Dense GCN (all edges)")
    dense_comp = measure_inference(
        dense_model, data, device, model_name="Dense GCN")

    # ------------------------------------------------------------------
    # Step 3: Pruning GNN
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("  STEP 3: Reconstruction-Aware Pruning GNN")
    print("=" * 60)

    pruning_model = PruningGNN(
        in_channels, args.ae_hidden, args.ae_latent,
        args.gnn_hidden, 2, args.dropout,
        target_sparsity=args.target_sparsity)

    print(f"  Training for {args.epochs} epochs...")
    print(f"  Temperature: {args.tau_start} -> {args.tau_end}")
    pruning_model, history = train_pruning_gnn(
        pruning_model, data, epochs=args.epochs, lr=args.lr,
        weight_decay=args.weight_decay,
        lambda1=args.lambda1, lambda2=args.lambda2,
        target_sparsity=args.target_sparsity,
        tau_start=args.tau_start, tau_end=args.tau_end,
        patience=args.patience,
        class_conditional_recon=args.class_conditional_recon,
        normal_class=0,
        device=device)

    pruned_metrics = evaluate_node_classification(
        pruning_model, data, data.test_mask, device,
        "Pruning GNN", is_pruning=True, tau=args.tau_end)
    pruned_comp = measure_inference(
        pruning_model, data, device, is_pruning=True,
        tau=args.tau_end, model_name="Pruning GNN")

    # Analyze noise edge targeting
    noise_analysis = analyze_noise_pruning(pruning_model, data, device, args.tau_end)

    # ------------------------------------------------------------------
    # Step 3b: Backbone-only inference on pre-pruned graph (deployment mode)
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("  STEP 3b: Backbone-Only Inference (Deployment Mode)")
    print("=" * 60)
    print("  Pre-compute pruning once, then run backbone on pruned graph.")

    # Extract pruned graph
    pruning_model.eval()
    with torch.no_grad():
        out = pruning_model(data.to(device).x, data.to(device).edge_index,
                            data.to(device).edge_attr, tau=args.tau_end)
        pruned_ei = out["pruned_edge_index"]
        pruned_ea = out["pruned_edge_attr"]

    # Create a pruned data object for backbone-only inference
    data_pruned = data.clone()
    data_pruned.edge_index = pruned_ei
    data_pruned.edge_attr = pruned_ea

    # Measure backbone-only on pruned graph
    backbone_only = pruning_model.backbone
    backbone_comp = measure_inference(
        backbone_only, data_pruned, device, model_name="Backbone on pruned graph")
    backbone_params = sum(p.numel() for p in backbone_only.parameters())

    # ------------------------------------------------------------------
    # Step 4: Summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("  FINAL COMPARISON")
    print("=" * 60)

    header = f"{'Metric':<20} {'MLP':>10} {'Dense GCN':>12} {'Pruning GNN':>14} {'Deploy Mode':>14}"
    print(header)
    print("-" * 60)

    for m in ["accuracy", "precision", "recall", "f1", "auc_roc"]:
        v1 = mlp_metrics[m]
        v2 = dense_metrics[m]
        v3 = pruned_metrics[m]
        print(f"  {m:<18} {v1:>10.4f} {v2:>12.4f} {v3:>14.4f} {v3:>14.4f}")

    print("-" * 60)
    dense_edges = dense_comp["total_edges"]
    pruned_edges = pruned_comp["kept_edges"]
    edge_reduction = 1.0 - pruned_edges / max(1, dense_edges)

    print(f"  {'Sparsity':<18} {'0%':>10} {'0%':>12} "
          f"{pruned_metrics['sparsity']:>13.2%} {pruned_metrics['sparsity']:>13.2%}")
    print(f"  {'Edge reduction':<18} {'—':>10} {'—':>12} "
          f"{edge_reduction:>13.2%} {edge_reduction:>13.2%}")
    print(f"  {'Parameters':<18} {mlp_comp['total_params']:>10,} {dense_comp['total_params']:>12,} "
          f"{pruned_comp['total_params']:>14,} {backbone_params:>14,}")
    print(f"  {'Inference (ms)':<18} {mlp_comp['inference_ms']:>10.2f} "
          f"{dense_comp['inference_ms']:>12.2f} {pruned_comp['inference_ms']:>14.2f} "
          f"{backbone_comp['inference_ms']:>14.2f}")

    print(f"\n  Key observations:")
    gcn_vs_mlp = dense_metrics["f1"] - mlp_metrics["f1"]
    prune_vs_dense = pruned_metrics["f1"] - dense_metrics["f1"]
    speedup = dense_comp["inference_ms"] / max(0.01, backbone_comp["inference_ms"])
    print(f"  - GCN vs MLP (graph value):      F1 {gcn_vs_mlp:+.4f}")
    print(f"  - Pruning vs Dense (prune value): F1 {prune_vs_dense:+.4f}")
    print(f"  - Edge reduction:                 {edge_reduction:.2%}")
    print(f"  - Deploy speedup vs Dense:        {speedup:.2f}x")
    print(f"  - Deploy params vs Dense:         {backbone_params:,} vs {dense_comp['total_params']:,} (same backbone)")

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------
    results = {
        "mlp_baseline": {**mlp_metrics, **mlp_comp},
        "dense_gcn": {**dense_metrics, **dense_comp},
        "pruning_gnn": {**pruned_metrics, **pruned_comp},
        "deploy_mode": {
            "inference_ms": backbone_comp["inference_ms"],
            "params": backbone_params,
            "kept_edges": pruned_comp["kept_edges"],
            "speedup_vs_dense": round(speedup, 4),
        },
        "noise_analysis": noise_analysis,
        "comparison": {
            "gcn_vs_mlp_f1": round(gcn_vs_mlp, 4),
            "prune_vs_dense_f1": round(prune_vs_dense, 4),
            "edge_reduction": round(edge_reduction, 4),
            "sparsity": pruned_metrics["sparsity"],
        },
        "training_history": {
            k: [round(v, 6) if isinstance(v, float) else v for v in vals]
            for k, vals in history.items()
        },
        "config": vars(args),
    }

    results_path = "results/pruning_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n[Pipeline] Results saved to {results_path}")
    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(
        description="Reconstruction-Aware Differentiable Pruning GNN for IoT IDS")

    # Data
    parser.add_argument("--num_nodes", type=int, default=5000)
    parser.add_argument("--edge_density", type=float, default=0.005)
    parser.add_argument("--attack_ratio", type=float, default=0.3)
    parser.add_argument("--feature_dim", type=int, default=16)
    parser.add_argument("--noise_edge_ratio", type=float, default=0.3,
                        help="Fraction of extra edges that are noisy cross-class")

    # Autoencoder
    parser.add_argument("--ae_hidden", type=int, default=64)
    parser.add_argument("--ae_latent", type=int, default=32)

    # GNN
    parser.add_argument("--gnn_hidden", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.5)

    # Training
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=5e-4)
    parser.add_argument("--patience", type=int, default=80)

    # Loss weights
    parser.add_argument("--lambda1", type=float, default=0.1)
    parser.add_argument("--lambda2", type=float, default=1.0,
                        help="Weight for sparsity constraint loss")
    parser.add_argument("--target_sparsity", type=float, default=0.3,
                        help="Target fraction of edges to prune")
    parser.add_argument("--class_conditional_recon", type=lambda x: x.lower() == "true",
                        default=True,
                        help="If True (default): AE recon loss masked to normal-class training nodes only "
                             "(self-supervised, label-efficient pruning signal). "
                             "If False: AE trained on all nodes (ablation).")

    # Temperature
    parser.add_argument("--tau_start", type=float, default=1.0)
    parser.add_argument("--tau_end", type=float, default=0.1)

    # General
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    main(args)
