# Session Context — Thesis Implementation

> This file captures the running state of the thesis implementation so it can
> be reconstructed across sessions. Update incrementally as work progresses.

## 1. Project at a glance

- **Title**: Self-Supervised Reconstruction-Guided Edge Pruning for GNN-based IoT Intrusion Detection
- **Goal**: Master's thesis showing that edge pruning works for IoT IDS, with a defensible novelty contribution and a clear gap argument.
- **Constraint**: Only Stage 1 (class-conditional autoencoder + Gumbel-Softmax pruner + 3-layer GCN). No model-level compression (Stage 2 deferred).

## 2. The novelty (single-sentence)

> Train an autoencoder only on **normal-class training nodes** so its per-node reconstruction error is an OOD score; feed that score into a differentiable Gumbel-Softmax edge pruner that runs before a supervised GCN classifier.

The architectural trick is in *training supervision*, not in the network topology:
- L_recon uses normal-class training nodes only (self-supervised; no attack labels)
- L_CE uses labeled training nodes (whatever attack labels exist)
- L_sparse uses no labels (just a target keep-probability)

This **decouples** the pruning signal from attack-label availability, addressing the label-asymmetry problem of real IoT IDS deployments.

## 3. The gap (one paragraph)

NeuralSparse, PTDNet, DropEdge, GNNGuard all drive their edge selection from the task gradient → they need attack labels for the entire data they prune. DOMINANT and AnomalyDAE use AE reconstruction at the node level, but they do not modify the graph. **Nobody combines the two**: nobody uses self-supervised reconstruction error as the input signal to a differentiable edge-pruning module inside a GNN classifier. That is the precise contribution.

## 4. Code map

| File | Purpose | Status |
|---|---|---|
| `node_level_data.py` | Synthetic IoT graph generator (configurable noise) | Stable |
| `reconstruction_pruning_model.py` | AutoencoderScorer, GumbelSoftmaxPruner, GNNBackbone, PruningGNN, tripartite_loss with `class_conditional_recon` toggle | Stable |
| `main_pipeline.py` | End-to-end training + comparison + deployment-mode timing | Stable; supports `--class_conditional_recon` flag |
| `label_efficiency_study.py` | Three-way comparison (Dense GCN, Ours, full-AE ablation) across attack-label fractions | Stable; uses class-weighted CE for fair comparison under imbalance |
| `THESIS_REPORT.md` | Markdown thesis report with results | Done |
| `THESIS.tex` (planned) | Overleaf LaTeX version with figures | To create |

## 5. Verified experimental results

### 5.1 Main comparison (single seed, noise=0.3, sparsity=50%)

| Metric | MLP | Dense GCN | Ours | Deploy mode |
|---|---|---|---|---|
| F1 | 0.685 | 0.995 | **0.995** | 0.995 |
| AUC-ROC | 0.873 | 1.000 | 1.000 | 1.000 |
| Edges | 130K | 130K | **67K (-48%)** | 67K |
| Inference (ms) | 0.5 | 7.6 | 11.8 (full) | **7.5** |

### 5.2 Selectivity (which edges get pruned)

- Noisy cross-class edges pruned: **63.7%**
- Clean same-class edges pruned: **44.8%**
- **Selectivity ratio: 1.42×** (ours preferentially targets noise)

### 5.3 Label-efficiency (5 seeds, mean ± std)

**Low-noise (noise_edge_ratio = 0.3)** — `results/label_efficiency_n03.json`

| Frac | Dense GCN F1 | Ours F1 | Full-AE F1 |
|---|---|---|---|
| 1.00 | 0.994 ± 0.002 | 0.983 ± 0.017 | 0.991 ± 0.003 |
| 0.50 | 0.994 ± 0.001 | 0.969 ± 0.024 | 0.989 ± 0.003 |
| 0.25 | 0.994 ± 0.002 | 0.977 ± 0.022 | 0.989 ± 0.003 |
| 0.10 | 0.993 ± 0.001 | 0.974 ± 0.017 | 0.988 ± 0.002 |
| 0.05 | 0.993 ± 0.000 | 0.972 ± 0.021 | 0.988 ± 0.003 |

**High-noise (noise_edge_ratio = 0.6)** — `results/label_efficiency_n06.json` (the novelty plot)

| Frac | Dense GCN F1 | Ours F1 | Full-AE F1 |
|---|---|---|---|
| 1.00 | 0.937 ± 0.005 | 0.797 ± 0.100 | 0.786 ± 0.126 |
| 0.50 | 0.934 ± 0.012 | 0.773 ± 0.083 | 0.781 ± 0.130 |
| 0.25 | 0.923 ± 0.020 | 0.765 ± 0.080 | 0.729 ± 0.112 |
| 0.10 | 0.901 ± 0.024 | 0.765 ± 0.076 | 0.721 ± 0.082 |
| 0.05 | **0.804 ± 0.178** | **0.755 ± 0.047** | 0.723 ± 0.071 |

The headline finding: **at 5% attack labels under high noise, ours has 3.8× lower std** than Dense GCN (0.047 vs 0.178), making it much more reliable for deployment even though its mean F1 is slightly lower.

### 5.4 Inference scaling

| Nodes | Edges | Dense (ms) | Pruned (ms) | Speedup |
|---|---|---|---|---|
| 5K | 130K | 44.8 | 44.1 | 1.01× |
| 10K | 410K | 46.1 | 44.7 | 1.03× |
| 20K | 820K | 52.8 | 46.2 | 1.14× |
| 50K | 2.05M | 74.7 | 54.7 | **1.37×** |

CUDA-synchronized timings, median of 100 runs after 30 warm-up.

## 6. Honest pitfalls / things to be careful about

1. **At noise=0.6 our absolute F1 (~0.79) is below Dense GCN's (~0.94)**. We trade accuracy for stability + edge reduction. Honest framing required.
2. **At noise=0.3 the label-efficiency story is muted** — Dense GCN doesn't really degrade, so "ours degrades less" is a small effect. The high-noise plot is what carries the novelty story.
3. **Class-weighted CE is needed** in the label-efficiency study (label imbalance causes Dense GCN to collapse otherwise). All three methods use it for fair comparison.
4. **The "full AE" ablation must use only training nodes** for the recon loss — using all nodes including test would be feature leakage. (Fixed: both branches now use `mask` and differ only in label condition.)

## 7. Decisions made (and why)

- **No scout GCN.** Adding it would deviate from the simple "Stage 1 only" architecture the user explicitly asked for. The simple Gumbel-Softmax pruner has weaker absolute F1 at high noise but a cleaner narrative.
- **No real dataset (TON_IoT) yet.** Synthetic gives ground-truth noisy edges (needed for selectivity ablation) and is fully controllable. Real-data validation is in "future work".
- **Target sparsity 50%.** Aggressive enough to show inference savings; not so high that F1 collapses.
- **5 seeds in label-efficiency study.** Enough to estimate variance; doubles wall-clock time of the experiment from ~6 min to ~30 min.

## 8. What is left to do (per user's latest request)

1. Convert thesis report to LaTeX (Overleaf-ready), comprehensive, ≤30 pages.
2. Include figures (architecture diagram, label-efficiency plots, selectivity bar, inference scaling).
3. Include tables for all results.
4. References as a `.bib` file.
5. Continue saving the context (this file).

## 9. Pointers for the next session

- The single most important plot for novelty defence: `results/label_efficiency_n06.png`
- The single most important number for novelty defence: **3.8× lower std at 5% attack labels under high noise**
- The single most important sentence for novelty defence: *"Our pruning module's loss path uses no attack labels."*
- All raw numbers live in `results/*.json`.
- The thesis report (Markdown) is the source of truth for narrative; LaTeX is generated from it.
