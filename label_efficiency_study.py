"""
Label-Efficiency Study: the key novelty experiment.
====================================================

This script tests how each method degrades as the fraction of labeled ATTACK
nodes shrinks. Normal labels are kept abundant (real IoT scenario).

Compared methods:
    1. Dense GCN — needs labels for backbone training
    2. Pruning GNN (class-conditional AE) — our method; AE doesn't need attack labels
    3. Pruning GNN (full AE, ablation) — AE trained on all nodes (uses attack labels)

Hypothesis: Method 2 degrades least because the pruning signal is independent
of attack-label availability.

Output:
    results/label_efficiency.json   (raw numbers per fraction × method × seed)
    results/label_efficiency.png    (line plot)
"""

import argparse
import copy
import json
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import f1_score, roc_auc_score

from node_level_data import generate_iot_graph
from reconstruction_pruning_model import (
    PruningGNN, GNNBackbone, tripartite_loss,
    NeuralSparseGNN, neuralsparse_loss, DropEdgeGNN,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def get_temperature(epoch, total_epochs, tau_start=1.0, tau_end=0.1):
    decay = (tau_end / tau_start) ** (epoch / max(1, total_epochs - 1))
    return tau_start * decay


def subsample_attack_labels(data, attack_label_fraction, seed):
    """
    Build a NEW train_mask that keeps:
      - all normal-class training nodes
      - only `attack_label_fraction` of the attack-class training nodes
    Validation and test masks are unchanged.
    """
    train_mask = data.train_mask.clone()
    y = data.y
    train_attack_idx = (train_mask & (y == 1)).nonzero(as_tuple=True)[0]
    n_keep = max(1, int(round(len(train_attack_idx) * attack_label_fraction)))

    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(len(train_attack_idx), generator=g)
    drop_idx = train_attack_idx[perm[n_keep:]]

    new_mask = train_mask.clone()
    new_mask[drop_idx] = False
    return new_mask, n_keep, len(train_attack_idx)


def class_weighted_ce(logits, y, mask):
    """Inverse-frequency weighted cross-entropy — important under severe label
    imbalance (e.g., 5% attack labels = 45:1 normal:attack ratio)."""
    y_train = y[mask]
    n_total = y_train.numel()
    counts = torch.bincount(y_train, minlength=2).float().clamp(min=1.0)
    weights = n_total / (2.0 * counts)  # inverse frequency, balanced to mean=1
    weights = weights.to(logits.device)
    return F.cross_entropy(logits[mask], y_train, weight=weights)


# ---------------------------------------------------------------------------
# Trainers
# ---------------------------------------------------------------------------
def train_dense_gcn(data, train_mask, in_channels, epochs=200, lr=1e-3,
                     wd=5e-4, patience=50, device="cuda"):
    model = GNNBackbone(in_channels, 128, 2, 0.5).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    best_f1, best_state, wait = 0.0, None, 0
    for ep in range(1, epochs + 1):
        model.train(); opt.zero_grad()
        logits = model(data.x, data.edge_index, edge_weight=data.edge_attr)
        class_weighted_ce(logits, data.y, train_mask).backward()
        opt.step()
        if ep % 5 == 0:
            model.eval()
            with torch.no_grad():
                vl = model(data.x, data.edge_index, edge_weight=data.edge_attr)
                vf = f1_score(data.y[data.val_mask].cpu(),
                              vl[data.val_mask].argmax(1).cpu(),
                              zero_division=0)
            if vf > best_f1:
                best_f1, best_state, wait = vf, copy.deepcopy(model.state_dict()), 0
            else:
                wait += 5
                if wait >= patience: break
    if best_state: model.load_state_dict(best_state)
    return model


def train_pruning_gnn_le(data, train_mask, in_channels,
                         class_conditional_recon=True,
                         target_sparsity=0.5, epochs=300, lr=1e-3, wd=5e-4,
                         lambda1=0.1, lambda2=1.0, warmup=50, patience=80,
                         device="cuda"):
    model = PruningGNN(in_channels, target_sparsity=target_sparsity).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    best_f1, best_state, wait = 0.0, None, 0
    prune_total = epochs - warmup
    for ep in range(1, epochs + 1):
        if ep <= warmup:
            tau, lam2 = 1.0, 0.0
        else:
            tau = get_temperature(ep - warmup, prune_total, 1.0, 0.1)
            lam2 = lambda2

        model.train(); opt.zero_grad()
        out = model(data.x, data.edge_index, data.edge_attr, tau=tau)

        # CE: class-weighted to handle severe label imbalance fairly
        l_ce = class_weighted_ce(out["logits"], data.y, train_mask)

        # Recon: class-conditional (ours) or full-train (ablation)
        if class_conditional_recon:
            recon_mask = train_mask & (data.y == 0)
        else:
            recon_mask = train_mask
        if recon_mask.sum() > 0:
            l_recon = F.mse_loss(out["x_recon"][recon_mask], data.x[recon_mask])
        else:
            l_recon = torch.tensor(0.0, device=data.x.device)

        # Sparsity term
        is_sl = (data.edge_index[0] == data.edge_index[1])
        non_loop = out["keep_probs"][~is_sl]
        if len(non_loop) > 0:
            l_sparse = (non_loop.mean() - (1.0 - target_sparsity)) ** 2
        else:
            l_sparse = torch.tensor(0.0, device=data.x.device)

        loss = l_ce + lambda1 * l_recon + lam2 * l_sparse
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step()

        if ep % 5 == 0:
            model.eval()
            with torch.no_grad():
                vo = model(data.x, data.edge_index, data.edge_attr, tau=tau)
                vf = f1_score(data.y[data.val_mask].cpu(),
                              vo["logits"][data.val_mask].argmax(1).cpu(),
                              zero_division=0)
            if vf > best_f1:
                best_f1, best_state, wait = vf, copy.deepcopy(model.state_dict()), 0
            else:
                wait += 5
                if wait >= patience and ep > warmup + 30: break
    if best_state: model.load_state_dict(best_state)
    return model


def train_neuralsparse(data, train_mask, in_channels,
                       target_sparsity=0.5, epochs=300, lr=1e-3, wd=5e-4,
                       lambda_sparse=1.0, warmup=50, patience=80,
                       device="cuda"):
    """NeuralSparse-style baseline: same Gumbel-Softmax pruner, but no AE
    and the pruner is trained from CE gradient only."""
    model = NeuralSparseGNN(in_channels, target_sparsity=target_sparsity).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    best_f1, best_state, wait = 0.0, None, 0
    prune_total = epochs - warmup
    for ep in range(1, epochs + 1):
        if ep <= warmup:
            tau, lam_s = 1.0, 0.0
        else:
            tau = get_temperature(ep - warmup, prune_total, 1.0, 0.1)
            lam_s = lambda_sparse

        model.train(); opt.zero_grad()
        out = model(data.x, data.edge_index, data.edge_attr, tau=tau)
        l_ce = class_weighted_ce(out["logits"], data.y, train_mask)
        is_sl = (data.edge_index[0] == data.edge_index[1])
        non_loop = out["keep_probs"][~is_sl]
        if len(non_loop) > 0:
            l_sparse = (non_loop.mean() - (1.0 - target_sparsity)) ** 2
        else:
            l_sparse = torch.tensor(0.0, device=data.x.device)
        loss = l_ce + lam_s * l_sparse
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step()

        if ep % 5 == 0:
            model.eval()
            with torch.no_grad():
                vo = model(data.x, data.edge_index, data.edge_attr, tau=tau)
                vf = f1_score(data.y[data.val_mask].cpu(),
                              vo["logits"][data.val_mask].argmax(1).cpu(),
                              zero_division=0)
            if vf > best_f1:
                best_f1, best_state, wait = vf, copy.deepcopy(model.state_dict()), 0
            else:
                wait += 5
                if wait >= patience and ep > warmup + 30: break
    if best_state: model.load_state_dict(best_state)
    return model


def train_dropedge(data, train_mask, in_channels, target_sparsity=0.5,
                    epochs=200, lr=1e-3, wd=5e-4, patience=50,
                    device="cuda", fixed_random=True):
    """DropEdge baseline (fixed random subset for fair edge-reduction comparison)."""
    model = DropEdgeGNN(in_channels, target_sparsity=target_sparsity,
                         fixed_random=fixed_random).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    best_f1, best_state, wait = 0.0, None, 0
    for ep in range(1, epochs + 1):
        model.train(); opt.zero_grad()
        out = model(data.x, data.edge_index, data.edge_attr)
        class_weighted_ce(out["logits"], data.y, train_mask).backward()
        opt.step()
        if ep % 5 == 0:
            model.eval()
            with torch.no_grad():
                vo = model(data.x, data.edge_index, data.edge_attr)
                vf = f1_score(data.y[data.val_mask].cpu(),
                              vo["logits"][data.val_mask].argmax(1).cpu(),
                              zero_division=0)
            if vf > best_f1:
                best_f1, best_state, wait = vf, copy.deepcopy(model.state_dict()), 0
            else:
                wait += 5
                if wait >= patience: break
    if best_state: model.load_state_dict(best_state)
    return model


def evaluate_pruning_no_tau(model, data, device):
    """Eval helper for models that don't use tau (DropEdge)."""
    model.eval()
    with torch.no_grad():
        out = model(data.x, data.edge_index, data.edge_attr)
    preds = out["logits"][data.test_mask].argmax(1).cpu().numpy()
    probs = F.softmax(out["logits"][data.test_mask], dim=1)[:, 1].cpu().numpy()
    yt = data.y[data.test_mask].cpu().numpy()
    return {
        "f1": f1_score(yt, preds, zero_division=0),
        "auc": roc_auc_score(yt, probs) if len(np.unique(yt)) > 1 else 0.0,
        "sparsity": out["sparsity"],
    }


def evaluate_dense(model, data, device, tau=None):
    model.eval()
    with torch.no_grad():
        logits = model(data.x, data.edge_index, edge_weight=data.edge_attr)
    preds = logits[data.test_mask].argmax(1).cpu().numpy()
    probs = F.softmax(logits[data.test_mask], dim=1)[:, 1].cpu().numpy()
    yt = data.y[data.test_mask].cpu().numpy()
    return {
        "f1": f1_score(yt, preds, zero_division=0),
        "auc": roc_auc_score(yt, probs) if len(np.unique(yt)) > 1 else 0.0,
    }


def evaluate_pruning(model, data, device, tau=0.1):
    model.eval()
    with torch.no_grad():
        out = model(data.x, data.edge_index, data.edge_attr, tau=tau)
    preds = out["logits"][data.test_mask].argmax(1).cpu().numpy()
    probs = F.softmax(out["logits"][data.test_mask], dim=1)[:, 1].cpu().numpy()
    yt = data.y[data.test_mask].cpu().numpy()
    return {
        "f1": f1_score(yt, preds, zero_division=0),
        "auc": roc_auc_score(yt, probs) if len(np.unique(yt)) > 1 else 0.0,
        "sparsity": out["sparsity"],
    }


# ---------------------------------------------------------------------------
# Main study
# ---------------------------------------------------------------------------
def run_study(args):
    device = args.device if torch.cuda.is_available() else "cpu"
    os.makedirs("results", exist_ok=True)

    # Build the graph ONCE (same graph across all conditions)
    base_data = generate_iot_graph(
        num_nodes=args.num_nodes,
        noise_edge_ratio=args.noise_edge_ratio,
        attack_ratio=args.attack_ratio,
        seed=args.data_seed,
    )
    in_channels = base_data.x.size(1)
    base_data = base_data.to(device)
    print(f"\n[LE Study] Graph: {args.num_nodes} nodes, "
          f"{base_data.edge_index.size(1):,} edges, "
          f"feature dim {in_channels}")
    print(f"[LE Study] Noise edge ratio: {args.noise_edge_ratio}, "
          f"target sparsity: {args.target_sparsity}")

    fractions = args.fractions
    seeds = list(range(args.n_seeds))
    methods = ["dense_gcn",
               "dropedge",
               "neuralsparse",
               "pruning_class_conditional",
               "pruning_full_ae"]

    all_records = []

    for frac in fractions:
        print("\n" + "=" * 70)
        print(f"  ATTACK-LABEL FRACTION = {frac:.0%}")
        print("=" * 70)
        for seed in seeds:
            torch.manual_seed(seed); np.random.seed(seed)
            if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

            train_mask, n_kept, n_total = subsample_attack_labels(
                base_data, frac, seed=seed)
            print(f"\n  [seed {seed}] using {n_kept}/{n_total} train attack labels")

            # ---- Method 1: Dense GCN ----
            t0 = time.time()
            m1 = train_dense_gcn(base_data, train_mask, in_channels,
                                 epochs=args.epochs_dense, device=device)
            r1 = evaluate_dense(m1, base_data, device)
            t1 = time.time() - t0
            print(f"    Dense GCN:                     F1={r1['f1']:.4f} | AUC={r1['auc']:.4f} ({t1:.1f}s)")
            all_records.append({"method": "dense_gcn", "fraction": frac,
                                "seed": seed, "n_attack_labels": n_kept, **r1})

            # ---- Method 2: Pruning + class-conditional AE (OURS) ----
            t0 = time.time()
            m2 = train_pruning_gnn_le(base_data, train_mask, in_channels,
                                      class_conditional_recon=True,
                                      target_sparsity=args.target_sparsity,
                                      epochs=args.epochs_pruning, device=device)
            r2 = evaluate_pruning(m2, base_data, device)
            t1 = time.time() - t0
            print(f"    Pruning (class-conditional):   F1={r2['f1']:.4f} | AUC={r2['auc']:.4f} | "
                  f"sparsity={r2['sparsity']:.2%} ({t1:.1f}s)")
            all_records.append({"method": "pruning_class_conditional",
                                "fraction": frac, "seed": seed,
                                "n_attack_labels": n_kept, **r2})

            # ---- Method 3: Pruning + full AE (ablation) ----
            t0 = time.time()
            m3 = train_pruning_gnn_le(base_data, train_mask, in_channels,
                                      class_conditional_recon=False,
                                      target_sparsity=args.target_sparsity,
                                      epochs=args.epochs_pruning, device=device)
            r3 = evaluate_pruning(m3, base_data, device)
            t1 = time.time() - t0
            print(f"    Pruning (full AE, ablation):   F1={r3['f1']:.4f} | AUC={r3['auc']:.4f} | "
                  f"sparsity={r3['sparsity']:.2%} ({t1:.1f}s)")
            all_records.append({"method": "pruning_full_ae",
                                "fraction": frac, "seed": seed,
                                "n_attack_labels": n_kept, **r3})

            # ---- Method 4: NeuralSparse (Zheng et al., ICML 2020) ----
            t0 = time.time()
            m4 = train_neuralsparse(base_data, train_mask, in_channels,
                                     target_sparsity=args.target_sparsity,
                                     epochs=args.epochs_pruning, device=device)
            r4 = evaluate_pruning(m4, base_data, device)
            t1 = time.time() - t0
            print(f"    NeuralSparse:                  F1={r4['f1']:.4f} | AUC={r4['auc']:.4f} | "
                  f"sparsity={r4['sparsity']:.2%} ({t1:.1f}s)")
            all_records.append({"method": "neuralsparse",
                                "fraction": frac, "seed": seed,
                                "n_attack_labels": n_kept, **r4})

            # ---- Method 5: DropEdge (Rong et al., ICLR 2020) ----
            t0 = time.time()
            m5 = train_dropedge(base_data, train_mask, in_channels,
                                 target_sparsity=args.target_sparsity,
                                 epochs=args.epochs_dense, device=device)
            r5 = evaluate_pruning_no_tau(m5, base_data, device)
            t1 = time.time() - t0
            print(f"    DropEdge (fixed-random):       F1={r5['f1']:.4f} | AUC={r5['auc']:.4f} | "
                  f"sparsity={r5['sparsity']:.2%} ({t1:.1f}s)")
            all_records.append({"method": "dropedge",
                                "fraction": frac, "seed": seed,
                                "n_attack_labels": n_kept, **r5})

    # ---- Aggregate ----
    print("\n" + "=" * 70)
    print("  AGGREGATED RESULTS (mean ± std over seeds)")
    print("=" * 70)
    summary = {}
    for method in methods:
        summary[method] = {}
        for frac in fractions:
            f1s = [r["f1"] for r in all_records
                   if r["method"] == method and r["fraction"] == frac]
            aucs = [r["auc"] for r in all_records
                    if r["method"] == method and r["fraction"] == frac]
            summary[method][f"{frac:.2f}"] = {
                "f1_mean": float(np.mean(f1s)), "f1_std": float(np.std(f1s)),
                "auc_mean": float(np.mean(aucs)), "auc_std": float(np.std(aucs)),
            }

    header = (f"\n  {'Fraction':>8} | {'Dense':>14} | {'DropEdge':>14} | "
              f"{'NeuralSparse':>14} | {'Ours (cls-cond)':>16} | {'Full AE':>14}")
    print(header)
    print("  " + "-" * (len(header) - 2))
    for frac in fractions:
        k = f"{frac:.2f}"
        row = f"  {frac:>8.2f}"
        for m in ["dense_gcn", "dropedge", "neuralsparse",
                  "pruning_class_conditional", "pruning_full_ae"]:
            s = summary[m][k]
            row += f" | {s['f1_mean']:>6.3f} ± {s['f1_std']:.3f}"
        print(row)

    # ---- Save ----
    out = {
        "config": {**vars(args), "fractions": fractions, "methods": methods},
        "records": all_records,
        "summary": summary,
    }
    with open("results/label_efficiency.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n[LE Study] Saved results to results/label_efficiency.json")

    # ---- Plot ----
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(8, 5))
        for method, label, marker in [
            ("dense_gcn", "Dense GCN (no pruning)", "s"),
            ("dropedge", "DropEdge (random pruning)", "x"),
            ("neuralsparse", "NeuralSparse (supervised pruning)", "D"),
            ("pruning_class_conditional", "Ours: class-conditional AE", "o"),
            ("pruning_full_ae", "Pruning + full AE (ablation)", "^"),
        ]:
            means = [summary[method][f"{f:.2f}"]["f1_mean"] for f in fractions]
            stds  = [summary[method][f"{f:.2f}"]["f1_std"] for f in fractions]
            ax.errorbar([f * 100 for f in fractions], means, yerr=stds,
                        marker=marker, label=label, capsize=3, linewidth=2)
        ax.set_xlabel("% of attack labels available during training")
        ax.set_ylabel("Test F1")
        ax.set_title("Label-Efficiency Study")
        ax.set_xscale("log")
        ax.set_xticks([f * 100 for f in fractions])
        ax.set_xticklabels([f"{int(f*100)}%" for f in fractions])
        ax.grid(True, alpha=0.3)
        ax.legend(loc="lower right")
        ax.set_ylim(0, 1.05)
        plt.tight_layout()
        plt.savefig("results/label_efficiency.png", dpi=140)
        print(f"[LE Study] Plot: results/label_efficiency.png")
    except Exception as e:
        print(f"[LE Study] Plot failed: {e}")

    return out


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--num_nodes", type=int, default=5000)
    p.add_argument("--noise_edge_ratio", type=float, default=0.3)
    p.add_argument("--attack_ratio", type=float, default=0.3)
    p.add_argument("--target_sparsity", type=float, default=0.5)
    p.add_argument("--epochs_dense", type=int, default=200)
    p.add_argument("--epochs_pruning", type=int, default=300)
    p.add_argument("--fractions", type=float, nargs="+",
                   default=[1.0, 0.5, 0.25, 0.1, 0.05])
    p.add_argument("--n_seeds", type=int, default=3)
    p.add_argument("--data_seed", type=int, default=42)
    p.add_argument("--device", type=str, default="cuda")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_study(args)
