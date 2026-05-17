"""
Reconstruction-Aware Differentiable Pruning GNN
================================================
Architecture:
  1. AutoencoderScorer: Lightweight AE computes MSE reconstruction error per node
  2. GumbelSoftmaxPruner: Converts edge features → binary edge masks via
     Gumbel-Softmax with Straight-Through (ST) estimator
  3. PruningGNN: 3-layer GCN backbone operating on the pruned graph → node classifier

Loss: L_total = L_CE + lambda1 * L_recon + lambda2 * L_sparse
  - L_CE:     Cross-entropy for node classification
  - L_recon:  MSE reconstruction error (autoencoder quality)
  - L_sparse: Target sparsity penalty on keep-probabilities

Key design:
  - Self-loops are always preserved (never pruned) to prevent NaN
  - Temperature annealing: tau goes from 1.0 → 0.1 over training
  - Gumbel-Softmax ST: hard argmax forward, soft gradients backward
  - At eval: hard top-K selection using learned scores
  - Pruned edges are REMOVED from edge_index at eval
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, BatchNorm
from torch_geometric.utils import add_self_loops


# ---------------------------------------------------------------------------
# 1. Autoencoder Scorer
# ---------------------------------------------------------------------------
class AutoencoderScorer(nn.Module):
    """
    Lightweight autoencoder that reconstructs node features.
    The MSE reconstruction error per node serves as an anomaly/saliency score.
    """

    def __init__(self, in_dim: int, hidden_dim: int = 64, latent_dim: int = 32):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, in_dim),
        )

    def forward(self, x: torch.Tensor):
        z = self.encoder(x)
        x_recon = self.decoder(z)
        mse_per_node = ((x - x_recon) ** 2).mean(dim=1)  # (N,)
        return x_recon, mse_per_node


# ---------------------------------------------------------------------------
# 2. Gumbel-Softmax Edge Pruner
# ---------------------------------------------------------------------------
def _sample_gumbel(shape, device, eps=1e-20):
    """Sample from Gumbel(0, 1)."""
    u = torch.rand(shape, device=device).clamp(eps, 1 - eps)
    return -torch.log(-torch.log(u))


class GumbelSoftmaxPruner(nn.Module):
    """
    Edge pruner using Gumbel-Softmax with Straight-Through estimator.

    Edge features: [mse_src, mse_dst, cosine_sim, edge_weight]
    → MLP → 2-class logits (keep vs prune) per edge
    → Gumbel-Softmax → soft/hard binary mask

    Training: Gumbel-Softmax with ST (hard forward, soft backward)
    Eval: hard top-K selection using learned scores
    Self-loops always preserved.
    """

    def __init__(self, in_channels: int, target_sparsity: float = 0.3):
        super().__init__()
        self.target_sparsity = target_sparsity
        # Edge scorer: [mse_src, mse_dst, cosine_sim, edge_weight] → 2-class logits
        self.score_net = nn.Sequential(
            nn.Linear(4, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 2),  # 2-class: [prune_logit, keep_logit]
        )

    def forward(self, x, mse_per_node, edge_index, edge_attr=None,
                tau=1.0, num_nodes=None):
        src, dst = edge_index[0], edge_index[1]
        is_self_loop = (src == dst)

        # Build edge features
        mse_src = mse_per_node[src]
        mse_dst = mse_per_node[dst]
        cos_sim = F.cosine_similarity(x[src], x[dst], dim=1)
        if edge_attr is not None:
            ew_norm = (edge_attr - edge_attr.mean()) / edge_attr.std().clamp(min=1e-6)
        else:
            ew_norm = torch.zeros_like(mse_src)

        edge_feats = torch.stack([mse_src, mse_dst, cos_sim, ew_norm], dim=1)
        logits_2c = self.score_net(edge_feats)  # (E, 2): [prune, keep]

        if self.training:
            # Gumbel-Softmax with Straight-Through estimator
            gumbel_noise = _sample_gumbel(logits_2c.shape, logits_2c.device)
            y_soft = F.softmax((logits_2c + gumbel_noise) / max(tau, 0.01), dim=1)

            # ST: hard argmax forward, soft gradient backward
            y_hard = torch.zeros_like(y_soft)
            y_hard.scatter_(1, y_soft.argmax(dim=1, keepdim=True), 1.0)
            y_gumbel = y_hard - y_soft.detach() + y_soft  # ST trick

            edge_mask = y_gumbel[:, 1]  # keep probability (index 1 = keep)
            keep_probs = y_soft[:, 1]   # soft keep prob for sparsity loss
        else:
            # Eval: hard top-K selection using keep logits
            keep_scores = logits_2c[:, 1] - logits_2c[:, 0]  # keep - prune

            non_loop = ~is_self_loop
            non_loop_scores = keep_scores.clone()
            non_loop_scores[is_self_loop] = float('inf')

            n_non_loop = non_loop.sum().item()
            n_keep = int(n_non_loop * (1.0 - self.target_sparsity))

            if n_keep > 0 and n_non_loop > 0:
                threshold = torch.topk(non_loop_scores[non_loop], n_keep).values[-1]
                edge_mask = (keep_scores >= threshold).float()
            else:
                edge_mask = torch.ones_like(keep_scores)

            keep_probs = torch.sigmoid(keep_scores)

        # Force self-loops to always be kept
        edge_mask = torch.where(is_self_loop, torch.ones_like(edge_mask), edge_mask)
        keep_probs = torch.where(is_self_loop, torch.ones_like(keep_probs), keep_probs)

        return edge_mask, keep_probs


# ---------------------------------------------------------------------------
# 3. GNN Backbone (3-layer GCN)
# ---------------------------------------------------------------------------
class GNNBackbone(nn.Module):
    """3-layer GCN: input → h1 → h2 → h3 → classifier."""

    def __init__(self, in_channels, hidden_channels=128, num_classes=2, dropout=0.5):
        super().__init__()
        h1 = hidden_channels // 2   # 64
        h2 = hidden_channels         # 128
        h3 = hidden_channels // 2   # 64

        self.conv1 = GCNConv(in_channels, h1)
        self.bn1 = BatchNorm(h1)
        self.conv2 = GCNConv(h1, h2)
        self.bn2 = BatchNorm(h2)
        self.conv3 = GCNConv(h2, h3)
        self.bn3 = BatchNorm(h3)

        self.classifier = nn.Sequential(
            nn.Linear(h3, h3 // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(h3 // 2, num_classes),
        )
        self.dropout = dropout

    def forward(self, x, edge_index, edge_weight=None):
        h = self.conv1(x, edge_index, edge_weight=edge_weight)
        h = self.bn1(h)
        h = F.relu(h)
        h = F.dropout(h, p=self.dropout, training=self.training)

        h = self.conv2(h, edge_index, edge_weight=edge_weight)
        h = self.bn2(h)
        h = F.relu(h)
        h = F.dropout(h, p=self.dropout, training=self.training)

        h = self.conv3(h, edge_index, edge_weight=edge_weight)
        h = self.bn3(h)
        h = F.relu(h)
        h = F.dropout(h, p=self.dropout, training=self.training)

        return self.classifier(h)


# ---------------------------------------------------------------------------
# 4. Full Pruning GNN Model
# ---------------------------------------------------------------------------
class PruningGNN(nn.Module):
    """
    Reconstruction-Aware Differentiable Pruning GNN.

    Forward:
      1. AutoencoderScorer → x_recon, mse_per_node
      2. GumbelSoftmaxPruner → edge_mask, keep_probs
      3. Apply mask (ST binary at train, hard removal at eval)
      4. GNNBackbone on pruned graph → node logits
    """

    def __init__(self, in_channels, ae_hidden=64, ae_latent=32,
                 gnn_hidden=128, num_classes=2, dropout=0.5,
                 target_sparsity=0.3):
        super().__init__()
        self.autoencoder = AutoencoderScorer(in_channels, ae_hidden, ae_latent)
        self.pruner = GumbelSoftmaxPruner(in_channels, target_sparsity=target_sparsity)
        self.backbone = GNNBackbone(in_channels, gnn_hidden, num_classes, dropout)

    def forward(self, x, edge_index, edge_attr=None, tau=1.0):
        num_nodes = x.size(0)

        # Step 1: Autoencoder scoring
        x_recon, mse_per_node = self.autoencoder(x)

        # Step 2: Gumbel-Softmax pruning mask
        edge_mask, keep_probs = self.pruner(
            x, mse_per_node, edge_index, edge_attr, tau=tau, num_nodes=num_nodes
        )

        # Step 3: Apply mask
        if self.training:
            # TRAINING: ST binary mask as edge weight multiplier
            # Gumbel-ST gives hard 0/1 forward but soft gradients backward
            if edge_attr is not None:
                weighted_attr = edge_attr * edge_mask
            else:
                weighted_attr = edge_mask
            pruned_edge_index = edge_index
            pruned_edge_attr = weighted_attr
        else:
            # EVAL: hard removal — physically remove pruned edges
            keep_idx = edge_mask.bool()
            pruned_edge_index = edge_index[:, keep_idx]
            pruned_edge_attr = edge_attr[keep_idx] if edge_attr is not None else None

        # Step 4: GNN backbone on pruned graph
        logits = self.backbone(x, pruned_edge_index, edge_weight=pruned_edge_attr)

        # Compute sparsity (excluding self-loops)
        is_self_loop = (edge_index[0] == edge_index[1])
        non_loop_mask = ~is_self_loop
        if non_loop_mask.sum() > 0:
            sparsity = 1.0 - edge_mask[non_loop_mask].mean().item()
        else:
            sparsity = 0.0

        return {
            "logits": logits,
            "x_recon": x_recon,
            "mse_per_node": mse_per_node,
            "edge_mask": edge_mask,
            "keep_probs": keep_probs,
            "sparsity": sparsity,
            "pruned_edge_index": pruned_edge_index,
            "pruned_edge_attr": pruned_edge_attr,
        }

    def forward_dense(self, x, edge_index, edge_attr=None):
        """Forward pass WITHOUT pruning (all edges kept). For baseline."""
        return self.backbone(x, edge_index, edge_weight=edge_attr)


# ---------------------------------------------------------------------------
# 5. Competitor baselines (NeuralSparse, DropEdge)
# ---------------------------------------------------------------------------
class NeuralSparseGNN(nn.Module):
    """
    NeuralSparse-style baseline (Zheng et al., ICML 2020).

    Same Gumbel-Softmax + Straight-Through edge pruner architecture as ours,
    but the edge scorer takes ONLY node-feature signals (no autoencoder, no
    reconstruction loss). The pruner is trained from the task-loss gradient
    only, which is precisely the supervision style we are arguing against
    in this thesis.

    Edge features: [x_src ⊕ x_dst, cosine_similarity, edge_weight]
    Loss: L_CE + lambda_sparse * L_sparse  (no L_recon)
    """

    def __init__(self, in_channels, gnn_hidden=128, num_classes=2,
                 dropout=0.5, target_sparsity=0.3):
        super().__init__()
        scorer_in = 2 * in_channels + 2  # [x_src, x_dst, cos_sim, edge_weight]
        self.scorer = nn.Sequential(
            nn.Linear(scorer_in, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 2),
        )
        self.target_sparsity = target_sparsity
        self.backbone = GNNBackbone(in_channels, gnn_hidden, num_classes,
                                     dropout)

    def _score_edges(self, x, edge_index, edge_attr):
        src, dst = edge_index[0], edge_index[1]
        cos_sim = F.cosine_similarity(x[src], x[dst], dim=1, eps=1e-8)
        if edge_attr is not None:
            ew = (edge_attr - edge_attr.mean()) / edge_attr.std().clamp(min=1e-6)
        else:
            ew = torch.zeros_like(cos_sim)
        feats = torch.cat([x[src], x[dst],
                           cos_sim.unsqueeze(1), ew.unsqueeze(1)], dim=1)
        return self.scorer(feats)  # (E, 2)

    def forward(self, x, edge_index, edge_attr=None, tau=1.0):
        logits_2c = self._score_edges(x, edge_index, edge_attr)
        src, dst = edge_index[0], edge_index[1]
        is_self_loop = (src == dst)

        if self.training:
            # Gumbel-Softmax with Straight-Through
            g = -torch.log(-torch.log(
                torch.rand(logits_2c.shape, device=logits_2c.device).clamp(1e-20, 1 - 1e-20)))
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

        # Force self-loops kept
        edge_mask = torch.where(is_self_loop, torch.ones_like(edge_mask), edge_mask)
        keep_probs = torch.where(is_self_loop, torch.ones_like(keep_probs), keep_probs)

        if self.training:
            weighted_attr = edge_attr * edge_mask if edge_attr is not None else edge_mask
            pruned_edge_index = edge_index
            pruned_edge_attr = weighted_attr
        else:
            keep_idx = edge_mask.bool()
            pruned_edge_index = edge_index[:, keep_idx]
            pruned_edge_attr = edge_attr[keep_idx] if edge_attr is not None else None

        out_logits = self.backbone(x, pruned_edge_index,
                                    edge_weight=pruned_edge_attr)

        non_loop_mask = ~is_self_loop
        sparsity = 1.0 - edge_mask[non_loop_mask].mean().item() if non_loop_mask.sum() > 0 else 0.0

        return {
            "logits": out_logits,
            "edge_mask": edge_mask,
            "keep_probs": keep_probs,
            "sparsity": sparsity,
            "pruned_edge_index": pruned_edge_index,
            "pruned_edge_attr": pruned_edge_attr,
        }


def neuralsparse_loss(logits, y, mask, keep_probs, edge_index,
                      lambda_sparse=1.0, target_sparsity=0.3, **kwargs):
    """L_CE + lambda_sparse * L_sparse  (no reconstruction)."""
    l_ce = F.cross_entropy(logits[mask], y[mask])
    is_self_loop = (edge_index[0] == edge_index[1])
    non_loop = keep_probs[~is_self_loop]
    if len(non_loop) > 0:
        l_sparse = (non_loop.mean() - (1.0 - target_sparsity)) ** 2
    else:
        l_sparse = torch.tensor(0.0, device=logits.device)
    return l_ce + lambda_sparse * l_sparse, l_ce, l_sparse


class DropEdgeGNN(nn.Module):
    """
    DropEdge baseline (Rong et al., ICLR 2020).

    Random Bernoulli edge dropout DURING TRAINING ONLY; full graph at eval.
    For a fair edge-reduction comparison, we provide both modes:
      - Standard DropEdge (regularization): drop_p edges dropped each epoch
        during training; full graph at eval (NO edge reduction).
      - Fixed-Random pruning: drop a fixed random subset; same edges removed
        for both training and eval (gives the same edge-reduction as ours).

    The fixed-random mode is what the paper-version of DropEdge calls
    'sub-graph sampling' and is the apples-to-apples competitor.
    """

    def __init__(self, in_channels, gnn_hidden=128, num_classes=2,
                 dropout=0.5, target_sparsity=0.3, fixed_random=True):
        super().__init__()
        self.target_sparsity = target_sparsity
        self.fixed_random = fixed_random
        self.backbone = GNNBackbone(in_channels, gnn_hidden, num_classes,
                                     dropout)
        self._fixed_mask = None  # set on first call when fixed_random=True

    def _ensure_fixed_mask(self, edge_index):
        if self._fixed_mask is None:
            E = edge_index.size(1)
            is_self_loop = (edge_index[0] == edge_index[1])
            mask = torch.ones(E, dtype=torch.bool, device=edge_index.device)
            non_loop_idx = (~is_self_loop).nonzero(as_tuple=True)[0]
            n_drop = int(non_loop_idx.numel() * self.target_sparsity)
            if n_drop > 0:
                perm = torch.randperm(non_loop_idx.numel(), device=edge_index.device)
                drop_idx = non_loop_idx[perm[:n_drop]]
                mask[drop_idx] = False
            self._fixed_mask = mask

    def forward(self, x, edge_index, edge_attr=None, **kwargs):
        is_self_loop = (edge_index[0] == edge_index[1])

        if self.fixed_random:
            # Same mask every call (deterministic edge subset)
            self._ensure_fixed_mask(edge_index)
            mask = self._fixed_mask
        else:
            # Stochastic per-call drop, only at training; full at eval
            if self.training:
                E = edge_index.size(1)
                drop = torch.rand(E, device=edge_index.device) < self.target_sparsity
                mask = ~drop | is_self_loop
            else:
                mask = torch.ones(edge_index.size(1), dtype=torch.bool,
                                   device=edge_index.device)

        pruned_edge_index = edge_index[:, mask]
        pruned_edge_attr = edge_attr[mask] if edge_attr is not None else None

        out_logits = self.backbone(x, pruned_edge_index,
                                    edge_weight=pruned_edge_attr)
        sparsity = 1.0 - mask[~is_self_loop].float().mean().item() if (~is_self_loop).sum() > 0 else 0.0
        return {
            "logits": out_logits,
            "sparsity": sparsity,
            "pruned_edge_index": pruned_edge_index,
            "pruned_edge_attr": pruned_edge_attr,
        }


# ---------------------------------------------------------------------------
# 5. Loss Functions
# ---------------------------------------------------------------------------
def tripartite_loss(logits, y, mask, x, x_recon, keep_probs, edge_index,
                    lambda1=0.1, lambda2=0.1, target_sparsity=0.3,
                    normal_class=0, class_conditional_recon=True,
                    **kwargs):
    """
    Tri-partite loss with class-conditional reconstruction:

        L_total = L_CE + lambda1 * L_recon + lambda2 * L_sparse

    Where:
      L_CE:     Cross-entropy over LABELED training nodes (supervised)
      L_recon:  MSE reconstruction error over NORMAL-class training nodes
                ONLY (self-supervised). Forces the autoencoder to learn the
                normal-traffic distribution; attack nodes will have high MSE
                (out-of-distribution score).
      L_sparse: (mean(keep_probs) - target_keep)^2 — pushes the average
                edge keep-probability toward the target sparsity. No labels.

    Setting `class_conditional_recon=False` recovers the previous behavior
    (ablation: AE trained on all nodes).
    """
    # ---- L_CE: supervised, on labeled training nodes ----
    l_ce = F.cross_entropy(logits[mask], y[mask])

    # ---- L_recon: SELF-SUPERVISED, on selected training nodes ----
    # IMPORTANT: in BOTH branches we restrict to training-set nodes only,
    # to avoid leaking test/val features through the AE. The branches differ
    # only in WHICH labels we use among training nodes.
    if class_conditional_recon:
        # OURS: train AE only on NORMAL-labeled training nodes
        # → AE learns the normal distribution
        # → MSE becomes an OOD score for attack nodes
        # → no attack labels used
        recon_node_mask = mask & (y == normal_class)
    else:
        # ABLATION: train AE on all training nodes (normal + attack)
        # → AE learns the mixed distribution
        # → MSE is no longer an OOD score for attacks
        # → attack labels are used (transitively)
        recon_node_mask = mask
    if recon_node_mask.sum() > 0:
        l_recon = F.mse_loss(x_recon[recon_node_mask], x[recon_node_mask])
    else:
        l_recon = torch.tensor(0.0, device=x.device)

    # ---- L_sparse: label-free, pushes avg keep_prob toward target ----
    is_self_loop = (edge_index[0] == edge_index[1])
    non_loop_probs = keep_probs[~is_self_loop]
    if len(non_loop_probs) > 0:
        target_keep = 1.0 - target_sparsity
        l_sparse = (non_loop_probs.mean() - target_keep) ** 2
    else:
        l_sparse = torch.tensor(0.0, device=x.device)

    total = l_ce + lambda1 * l_recon + lambda2 * l_sparse
    return total, l_ce, l_recon, l_sparse
