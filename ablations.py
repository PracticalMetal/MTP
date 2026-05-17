"""
Ablation experiments for the thesis defence.

Runs:
    A1 — Pretrained-AE → NeuralSparse: shows that pre-training the AE and
         feeding its OOD scores as features to an otherwise-NeuralSparse pruner
         does NOT recover the flat-curve property. Closes the most likely
         panel attack: 'did you try just pretraining the AE separately?'

    A2 — Target sparsity sweep (rho in {0.2, 0.3, 0.5, 0.7}).
    A3 — Loss-weight (lambda_1, lambda_2) sensitivity.
    A4 — AE latent-dimension sweep (16, 32, 64).

Each ablation runs on the SAME synthetic graph (seed 42, noise_edge_ratio
chosen to give the discriminative regime) for fair comparison.

Output:
    results/ablation_A1.json  ... results/ablation_A4.json
    results/ablation_log.txt
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
    PruningGNN, GNNBackbone, AutoencoderScorer, tripartite_loss,
    NeuralSparseGNN,
)


# ---------------------------------------------------------------------------
# Shared utilities
# ---------------------------------------------------------------------------
def get_temperature(epoch, total_epochs, tau_start=1.0, tau_end=0.1):
    decay = (tau_end / tau_start) ** (epoch / max(1, total_epochs - 1))
    return tau_start * decay


def class_weighted_ce(logits, y, mask):
    y_train = y[mask]
    n_total = y_train.numel()
    counts = torch.bincount(y_train, minlength=2).float().clamp(min=1.0)
    weights = (n_total / (2.0 * counts)).to(logits.device)
    return F.cross_entropy(logits[mask], y_train, weight=weights)


def subsample_attack_labels(data, attack_label_fraction, seed):
    train_mask = data.train_mask.clone()
    y = data.y
    train_attack_idx = (train_mask & (y == 1)).nonzero(as_tuple=True)[0]
    n_keep = max(1, int(round(len(train_attack_idx) * attack_label_fraction)))
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(len(train_attack_idx), generator=g)
    drop_idx = train_attack_idx[perm[n_keep:]]
    new_mask = train_mask.clone()
    new_mask[drop_idx] = False
    return new_mask, n_keep


def evaluate(model, data, fn_name="pruning"):
    model.eval()
    with torch.no_grad():
        if fn_name == "pruning":
            out = model(data.x, data.edge_index, data.edge_attr, tau=0.1)
            logits = out["logits"]
            sparsity = out["sparsity"]
        elif fn_name == "neuralsparse":
            out = model(data.x, data.edge_index, data.edge_attr, tau=0.1)
            logits = out["logits"]
            sparsity = out["sparsity"]
        else:
            logits = model(data.x, data.edge_index, edge_weight=data.edge_attr)
            sparsity = 0.0
    preds = logits[data.test_mask].argmax(1).cpu().numpy()
    probs = F.softmax(logits[data.test_mask], dim=1)[:, 1].cpu().numpy()
    yt = data.y[data.test_mask].cpu().numpy()
    return {
        "f1": float(f1_score(yt, preds, zero_division=0)),
        "auc": float(roc_auc_score(yt, probs)) if len(np.unique(yt)) > 1 else 0.0,
        "sparsity": float(sparsity),
    }


# ---------------------------------------------------------------------------
# Pre-trained-AE → NeuralSparse model (the A1 ablation)
# ---------------------------------------------------------------------------
class NeuralSparseWithAEFeatures(nn.Module):
    """
    NeuralSparse architecture, but the edge scorer takes AE-derived
    OOD scores (MSE_u, MSE_v) as additional features. The AE is
    pre-trained on normal-class nodes only and then FROZEN.

    Loss is still purely classification (no L_recon). This isolates
    whether 'just having OOD features' is enough, or whether joint
    AE training is necessary.
    """

    def __init__(self, in_channels, frozen_ae, target_sparsity=0.5,
                 gnn_hidden=128, num_classes=2, dropout=0.5):
        super().__init__()
        self.frozen_ae = frozen_ae
        for p in self.frozen_ae.parameters():
            p.requires_grad = False

        # Same edge feature set as our method: [mse_u, mse_v, cos, w]
        self.scorer = nn.Sequential(
            nn.Linear(4, 32), nn.ReLU(),
            nn.Linear(32, 16), nn.ReLU(),
            nn.Linear(16, 2),
        )
        self.target_sparsity = target_sparsity
        self.backbone = GNNBackbone(in_channels, gnn_hidden, num_classes, dropout)

    def _score_edges(self, x, edge_index, edge_attr):
        src, dst = edge_index[0], edge_index[1]
        # Frozen AE — same OOD signal as our method
        with torch.no_grad():
            _, mse = self.frozen_ae(x)
        cos_sim = F.cosine_similarity(x[src], x[dst], dim=1, eps=1e-8)
        if edge_attr is not None:
            ew = (edge_attr - edge_attr.mean()) / edge_attr.std().clamp(min=1e-6)
        else:
            ew = torch.zeros_like(cos_sim)
        feats = torch.stack([mse[src], mse[dst], cos_sim, ew], dim=1)
        return self.scorer(feats)

    def forward(self, x, edge_index, edge_attr=None, tau=1.0):
        logits_2c = self._score_edges(x, edge_index, edge_attr)
        src, dst = edge_index[0], edge_index[1]
        is_self_loop = (src == dst)

        if self.training:
            g = -torch.log(-torch.log(torch.rand_like(logits_2c).clamp(1e-20, 1 - 1e-20)))
            y_soft = F.softmax((logits_2c + g) / max(tau, 0.01), dim=1)
            y_hard = torch.zeros_like(y_soft)
            y_hard.scatter_(1, y_soft.argmax(dim=1, keepdim=True), 1.0)
            y_st = y_hard - y_soft.detach() + y_soft
            edge_mask = y_st[:, 1]
            keep_probs = y_soft[:, 1]
        else:
            keep_score = logits_2c[:, 1] - logits_2c[:, 0]
            non_loop = ~is_self_loop
            non_loop_scores = keep_score.clone()
            non_loop_scores[is_self_loop] = float('inf')
            n_keep = int(non_loop.sum().item() * (1.0 - self.target_sparsity))
            if n_keep > 0 and non_loop.sum() > 0:
                threshold = torch.topk(non_loop_scores[non_loop], n_keep).values[-1]
                edge_mask = (keep_score >= threshold).float()
            else:
                edge_mask = torch.ones_like(keep_score)
            keep_probs = torch.sigmoid(keep_score)

        edge_mask = torch.where(is_self_loop, torch.ones_like(edge_mask), edge_mask)
        keep_probs = torch.where(is_self_loop, torch.ones_like(keep_probs), keep_probs)

        if self.training:
            weighted_attr = edge_attr * edge_mask if edge_attr is not None else edge_mask
            pei = edge_index
            pea = weighted_attr
        else:
            keep_idx = edge_mask.bool()
            pei = edge_index[:, keep_idx]
            pea = edge_attr[keep_idx] if edge_attr is not None else None

        out_logits = self.backbone(x, pei, edge_weight=pea)
        non_loop_mask = ~is_self_loop
        sparsity = 1.0 - edge_mask[non_loop_mask].mean().item() if non_loop_mask.sum() > 0 else 0.0
        return {"logits": out_logits, "edge_mask": edge_mask, "keep_probs": keep_probs,
                "sparsity": sparsity, "pruned_edge_index": pei, "pruned_edge_attr": pea}


def pretrain_ae_on_normals(data, in_channels, train_mask, epochs=100, lr=1e-3, device="cuda"):
    """Pre-train AE on normal-class training nodes only."""
    ae = AutoencoderScorer(in_channels, 64, 32).to(device)
    opt = torch.optim.Adam(ae.parameters(), lr=lr, weight_decay=5e-4)
    normal_mask = train_mask & (data.y == 0)
    for ep in range(epochs):
        ae.train(); opt.zero_grad()
        x_recon, _ = ae(data.x)
        if normal_mask.sum() > 0:
            loss = F.mse_loss(x_recon[normal_mask], data.x[normal_mask])
            loss.backward(); opt.step()
    ae.eval()
    return ae


# ---------------------------------------------------------------------------
# Trainers (compact)
# ---------------------------------------------------------------------------
def train_pruning(data, train_mask, in_channels,
                   target_sparsity=0.5, epochs=300, lr=1e-3, wd=5e-4,
                   lambda1=0.1, lambda2=1.0, warmup=50, patience=80,
                   class_conditional=True, ae_latent=32, device="cuda"):
    model = PruningGNN(in_channels, ae_latent=ae_latent,
                       target_sparsity=target_sparsity).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    best_f1, best_state, wait = 0.0, None, 0
    prune_total = epochs - warmup
    for ep in range(1, epochs + 1):
        if ep <= warmup:
            tau, lam_s = 1.0, 0.0
        else:
            tau = get_temperature(ep - warmup, prune_total, 1.0, 0.1)
            lam_s = lambda2
        model.train(); opt.zero_grad()
        out = model(data.x, data.edge_index, data.edge_attr, tau=tau)
        l_ce = class_weighted_ce(out["logits"], data.y, train_mask)
        if class_conditional:
            recon_mask = train_mask & (data.y == 0)
        else:
            recon_mask = train_mask
        if recon_mask.sum() > 0:
            l_recon = F.mse_loss(out["x_recon"][recon_mask], data.x[recon_mask])
        else:
            l_recon = torch.tensor(0.0, device=data.x.device)
        is_sl = (data.edge_index[0] == data.edge_index[1])
        non_loop = out["keep_probs"][~is_sl]
        l_sparse = (non_loop.mean() - (1.0 - target_sparsity)) ** 2 if len(non_loop) > 0 else torch.tensor(0.0, device=data.x.device)
        loss = l_ce + lambda1 * l_recon + lam_s * l_sparse
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


def train_ae_plus_neuralsparse(data, train_mask, in_channels,
                                target_sparsity=0.5, epochs=300, lr=1e-3, wd=5e-4,
                                warmup=50, patience=80, device="cuda"):
    """A1 ablation: pre-train AE on normals, then train NeuralSparse-style
    pruner using AE features. Loss is L_CE only (no L_recon)."""
    # Phase 1: pre-train AE
    frozen_ae = pretrain_ae_on_normals(data, in_channels, train_mask, epochs=100, device=device)
    # Phase 2: train NeuralSparse with AE features
    model = NeuralSparseWithAEFeatures(in_channels, frozen_ae, target_sparsity=target_sparsity).to(device)
    # Only optimise scorer + backbone (AE is frozen)
    trainable = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.Adam(trainable, lr=lr, weight_decay=wd)
    best_f1, best_state, wait = 0.0, None, 0
    prune_total = epochs - warmup
    for ep in range(1, epochs + 1):
        tau = 1.0 if ep <= warmup else get_temperature(ep - warmup, prune_total, 1.0, 0.1)
        lam_s = 0.0 if ep <= warmup else 1.0
        model.train(); opt.zero_grad()
        out = model(data.x, data.edge_index, data.edge_attr, tau=tau)
        l_ce = class_weighted_ce(out["logits"], data.y, train_mask)
        is_sl = (data.edge_index[0] == data.edge_index[1])
        non_loop = out["keep_probs"][~is_sl]
        l_sparse = (non_loop.mean() - (1.0 - target_sparsity)) ** 2 if len(non_loop) > 0 else torch.tensor(0.0, device=data.x.device)
        loss = l_ce + lam_s * l_sparse
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, 5.0)
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


# ---------------------------------------------------------------------------
# A1: Pretrained-AE → NeuralSparse vs Ours
# ---------------------------------------------------------------------------
def run_A1(args, base_data, in_channels, device):
    print("\n" + "=" * 60)
    print("  A1: Pretrained-AE → NeuralSparse (closes Attack 1)")
    print("=" * 60)
    fractions = [1.0, 0.5, 0.25, 0.1, 0.05]
    seeds = list(range(3))
    records = []
    for frac in fractions:
        for seed in seeds:
            torch.manual_seed(seed); np.random.seed(seed)
            if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
            train_mask, n_kept = subsample_attack_labels(base_data, frac, seed=seed)
            # Ours
            t0 = time.time()
            m_ours = train_pruning(base_data, train_mask, in_channels,
                                    target_sparsity=0.5, device=device)
            r_ours = evaluate(m_ours, base_data, "pruning")
            # Pretrained-AE → NeuralSparse
            t1 = time.time()
            m_ae_ns = train_ae_plus_neuralsparse(base_data, train_mask, in_channels,
                                                  target_sparsity=0.5, device=device)
            r_ae_ns = evaluate(m_ae_ns, base_data, "neuralsparse")
            t2 = time.time()
            print(f"  frac={frac:.2f} seed={seed} | "
                  f"Ours F1={r_ours['f1']:.4f} ({t1-t0:.0f}s) | "
                  f"PretrainedAE→NS F1={r_ae_ns['f1']:.4f} ({t2-t1:.0f}s)")
            records.append({"frac": frac, "seed": seed,
                            "ours": r_ours, "pretrained_ae_neuralsparse": r_ae_ns})
    # Aggregate
    summary = {}
    for method in ["ours", "pretrained_ae_neuralsparse"]:
        summary[method] = {}
        for frac in fractions:
            f1s = [r[method]["f1"] for r in records if r["frac"] == frac]
            summary[method][f"{frac:.2f}"] = {
                "f1_mean": float(np.mean(f1s)), "f1_std": float(np.std(f1s)),
            }
    print("\n  Summary (3 seeds, mean ± std):")
    print(f"  {'Fraction':>10} | {'Ours F1':>18} | {'PretrainedAE→NS F1':>22}")
    for frac in fractions:
        k = f"{frac:.2f}"
        print(f"  {frac:>10.2f} | "
              f"{summary['ours'][k]['f1_mean']:>8.4f} ± {summary['ours'][k]['f1_std']:.4f} | "
              f"{summary['pretrained_ae_neuralsparse'][k]['f1_mean']:>10.4f} ± {summary['pretrained_ae_neuralsparse'][k]['f1_std']:.4f}")
    out = {"records": records, "summary": summary}
    with open("results/ablation_A1.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"  Saved → results/ablation_A1.json")
    return summary


# ---------------------------------------------------------------------------
# A2: Target sparsity sweep
# ---------------------------------------------------------------------------
def run_A2(args, base_data, in_channels, device):
    print("\n" + "=" * 60)
    print("  A2: Target sparsity sweep")
    print("=" * 60)
    sparsities = [0.2, 0.3, 0.5, 0.7]
    seeds = list(range(3))
    records = []
    for rho in sparsities:
        for seed in seeds:
            torch.manual_seed(seed); np.random.seed(seed)
            if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
            train_mask = base_data.train_mask.clone()
            t0 = time.time()
            m = train_pruning(base_data, train_mask, in_channels,
                               target_sparsity=rho, device=device)
            r = evaluate(m, base_data, "pruning")
            t1 = time.time()
            print(f"  rho={rho:.2f} seed={seed} | F1={r['f1']:.4f} | "
                  f"AUC={r['auc']:.4f} | sparsity={r['sparsity']:.2%} ({t1-t0:.0f}s)")
            records.append({"rho": rho, "seed": seed, **r})
    summary = {}
    for rho in sparsities:
        rs = [r for r in records if r["rho"] == rho]
        summary[f"{rho:.2f}"] = {
            "f1_mean": float(np.mean([r["f1"] for r in rs])),
            "f1_std":  float(np.std([r["f1"] for r in rs])),
            "auc_mean": float(np.mean([r["auc"] for r in rs])),
            "sparsity_mean": float(np.mean([r["sparsity"] for r in rs])),
        }
    print("\n  Summary:")
    for rho in sparsities:
        k = f"{rho:.2f}"
        s = summary[k]
        print(f"  rho={rho:.2f} | F1={s['f1_mean']:.4f} ± {s['f1_std']:.4f} | "
              f"AUC={s['auc_mean']:.4f} | achieved sparsity={s['sparsity_mean']:.2%}")
    out = {"records": records, "summary": summary}
    with open("results/ablation_A2.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"  Saved → results/ablation_A2.json")
    return summary


# ---------------------------------------------------------------------------
# A3: lambda1 / lambda2 sensitivity
# ---------------------------------------------------------------------------
def run_A3(args, base_data, in_channels, device):
    print("\n" + "=" * 60)
    print("  A3: Loss-weight (lambda_1, lambda_2) sensitivity")
    print("=" * 60)
    grid = [
        (0.01, 1.0), (0.1, 1.0), (1.0, 1.0),    # lambda1 sweep
        (0.1, 0.1),  (0.1, 0.5), (0.1, 2.0),     # lambda2 sweep
    ]
    seeds = list(range(3))
    records = []
    for lam1, lam2 in grid:
        for seed in seeds:
            torch.manual_seed(seed); np.random.seed(seed)
            if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
            train_mask = base_data.train_mask.clone()
            t0 = time.time()
            m = train_pruning(base_data, train_mask, in_channels,
                               lambda1=lam1, lambda2=lam2,
                               target_sparsity=0.5, device=device)
            r = evaluate(m, base_data, "pruning")
            t1 = time.time()
            print(f"  l1={lam1:.2f} l2={lam2:.2f} seed={seed} | F1={r['f1']:.4f} | "
                  f"sparsity={r['sparsity']:.2%} ({t1-t0:.0f}s)")
            records.append({"lambda1": lam1, "lambda2": lam2, "seed": seed, **r})
    summary = {}
    for lam1, lam2 in grid:
        key = f"l1={lam1}_l2={lam2}"
        rs = [r for r in records if r["lambda1"] == lam1 and r["lambda2"] == lam2]
        summary[key] = {
            "f1_mean": float(np.mean([r["f1"] for r in rs])),
            "f1_std":  float(np.std([r["f1"] for r in rs])),
            "sparsity_mean": float(np.mean([r["sparsity"] for r in rs])),
        }
    print("\n  Summary:")
    for k, s in summary.items():
        print(f"  {k:>20s} | F1={s['f1_mean']:.4f} ± {s['f1_std']:.4f} | "
              f"sparsity={s['sparsity_mean']:.2%}")
    out = {"records": records, "summary": summary}
    with open("results/ablation_A3.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"  Saved → results/ablation_A3.json")
    return summary


# ---------------------------------------------------------------------------
# A4: AE latent-dim sweep
# ---------------------------------------------------------------------------
def run_A4(args, base_data, in_channels, device):
    print("\n" + "=" * 60)
    print("  A4: AE latent-dim sweep")
    print("=" * 60)
    latents = [16, 32, 64]
    seeds = list(range(3))
    records = []
    for ld in latents:
        for seed in seeds:
            torch.manual_seed(seed); np.random.seed(seed)
            if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
            train_mask = base_data.train_mask.clone()
            t0 = time.time()
            m = train_pruning(base_data, train_mask, in_channels,
                               ae_latent=ld, target_sparsity=0.5, device=device)
            r = evaluate(m, base_data, "pruning")
            t1 = time.time()
            print(f"  latent={ld:>3d} seed={seed} | F1={r['f1']:.4f} | "
                  f"sparsity={r['sparsity']:.2%} ({t1-t0:.0f}s)")
            records.append({"latent": ld, "seed": seed, **r})
    summary = {}
    for ld in latents:
        rs = [r for r in records if r["latent"] == ld]
        summary[str(ld)] = {
            "f1_mean": float(np.mean([r["f1"] for r in rs])),
            "f1_std":  float(np.std([r["f1"] for r in rs])),
        }
    print("\n  Summary:")
    for ld in latents:
        s = summary[str(ld)]
        print(f"  latent={ld:>3d} | F1={s['f1_mean']:.4f} ± {s['f1_std']:.4f}")
    out = {"records": records, "summary": summary}
    with open("results/ablation_A4.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"  Saved → results/ablation_A4.json")
    return summary


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--num_nodes", type=int, default=5000)
    p.add_argument("--noise_edge_ratio", type=float, default=0.6,
                   help="0.6 is the discriminative regime")
    p.add_argument("--data_seed", type=int, default=42)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--run", type=str, default="all",
                   choices=["all", "A1", "A2", "A3", "A4"])
    return p.parse_args()


def main():
    args = parse_args()
    device = args.device if torch.cuda.is_available() else "cpu"
    os.makedirs("results", exist_ok=True)
    base_data = generate_iot_graph(
        num_nodes=args.num_nodes,
        noise_edge_ratio=args.noise_edge_ratio,
        attack_ratio=0.3,
        seed=args.data_seed,
    ).to(device)
    in_channels = base_data.x.size(1)
    print(f"[ablations] graph: {args.num_nodes} nodes, "
          f"{base_data.edge_index.size(1):,} edges, "
          f"noise_ratio={args.noise_edge_ratio}")

    results = {"config": vars(args)}
    if args.run in ("all", "A1"): results["A1"] = run_A1(args, base_data, in_channels, device)
    if args.run in ("all", "A2"): results["A2"] = run_A2(args, base_data, in_channels, device)
    if args.run in ("all", "A3"): results["A3"] = run_A3(args, base_data, in_channels, device)
    if args.run in ("all", "A4"): results["A4"] = run_A4(args, base_data, in_channels, device)

    with open("results/ablation_all.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\n[ablations] All done. Combined results → results/ablation_all.json")


if __name__ == "__main__":
    main()
