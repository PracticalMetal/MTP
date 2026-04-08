"""
SLIDE: SLImming DE-correlation Fine-tuning
============================================
Neural network parameter pruning for GNN compression.

Two-step process:
  Step 1 (Neuron Pruning): Randomly prune 30–50% of hidden neurons from
    pre-trained GCN layers. Uses magnitude-based selection combined with
    random masking to maintain diversity.

  Step 2 (De-correlation Fine-tuning): Fine-tune the pruned model with a
    combined loss:
      L = L_CE + λ * L_decorr
    where L_decorr is the squared Frobenius norm of the reweighted partial
    cross-covariance matrix estimated via Random Fourier Features (RFF).

    The de-correlation penalty minimizes embedding redundancy among the
    surviving neurons, forcing them to encode diverse information.

Reference:
  - Liu et al., "Rethinking the Value of Network Pruning," ICLR 2019.
  - Chen et al., "SLIDE: De-correlation Fine-tuning for Slimming Neural
    Networks," NeurIPS 2021.
  - Rahimi & Recht, "Random Features for Large-Scale Kernel Machines,"
    NeurIPS 2007 (RFF).
"""

import copy
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


# ---------------------------------------------------------------------------
# Random Fourier Features for de-correlation
# ---------------------------------------------------------------------------

class RandomFourierFeatures(nn.Module):
    """
    Approximate RBF kernel features via random Fourier projections.

    φ(x) = sqrt(2/D) * cos(Wx + b)

    where W ~ N(0, 1/σ²), b ~ Uniform(0, 2π).

    Reference: Rahimi & Recht, "Random Features for Large-Scale Kernel
    Machines," NeurIPS 2007.

    Parameters
    ----------
    in_dim : int
        Input feature dimension.
    rff_dim : int
        Number of random Fourier features (output dimension).
    sigma : float
        Bandwidth of the RBF kernel.
    """

    def __init__(self, in_dim: int, rff_dim: int = 256, sigma: float = 1.0):
        super().__init__()
        self.rff_dim = rff_dim
        # Fixed random projection (not learnable)
        self.register_buffer("W", torch.randn(in_dim, rff_dim) / sigma)
        self.register_buffer("b", torch.rand(rff_dim) * 2 * math.pi)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : Tensor (batch, in_dim)

        Returns
        -------
        phi : Tensor (batch, rff_dim) — RFF-approximated kernel features.
        """
        proj = x @ self.W + self.b  # (batch, rff_dim)
        phi = math.sqrt(2.0 / self.rff_dim) * torch.cos(proj)
        return phi


# ---------------------------------------------------------------------------
# De-correlation Loss
# ---------------------------------------------------------------------------

def decorrelation_loss(embeddings: torch.Tensor, rff_module: RandomFourierFeatures) -> torch.Tensor:
    """
    Compute the de-correlation penalty as the squared Frobenius norm of the
    reweighted partial cross-covariance matrix of node embeddings.

    L_decorr = ||C||²_F

    where C is the cross-covariance between the RFF-transformed embeddings
    of different neurons (feature dimensions).

    This encourages surviving neurons to encode non-redundant information.

    Reference: Chen et al., "SLIDE," NeurIPS 2021 — Equation (5).

    Parameters
    ----------
    embeddings : Tensor (num_nodes, hidden_dim)
        Node embeddings from pruned GCN layers.
    rff_module : RandomFourierFeatures
        RFF module for kernel approximation.

    Returns
    -------
    loss : scalar Tensor — squared Frobenius norm of cross-covariance.
    """
    # Apply RFF to each feature dimension treated as a sample
    # embeddings: (N, D) where N = nodes, D = hidden neurons
    N, D = embeddings.shape

    if N < 2 or D < 2:
        return torch.tensor(0.0, device=embeddings.device, requires_grad=True)

    # Center embeddings
    emb_centered = embeddings - embeddings.mean(dim=0, keepdim=True)

    # Cross-covariance matrix: C = (1/N) * emb^T @ emb
    # This is D x D — measures correlation between neurons
    C = (emb_centered.T @ emb_centered) / N  # (D, D)

    # Zero out diagonal (we only penalize off-diagonal correlations)
    mask = 1.0 - torch.eye(D, device=C.device)
    C_off = C * mask

    # Squared Frobenius norm
    loss = torch.sum(C_off ** 2)

    # Also add RFF-based penalty for nonlinear decorrelation
    # Transform embeddings through RFF and compute cross-covariance
    phi = rff_module(embeddings)  # (N, rff_dim)
    phi_centered = phi - phi.mean(dim=0, keepdim=True)

    # Cross-covariance in RFF space
    rff_dim = phi.shape[1]
    C_rff = (phi_centered.T @ phi_centered) / N  # (rff_dim, rff_dim)
    mask_rff = 1.0 - torch.eye(rff_dim, device=C_rff.device)
    C_rff_off = C_rff * mask_rff

    loss_rff = torch.sum(C_rff_off ** 2) / (rff_dim ** 2)  # normalize by dimension

    return loss + loss_rff


# ---------------------------------------------------------------------------
# Neuron Pruning Mask
# ---------------------------------------------------------------------------

def create_pruning_masks(
    model: nn.Module,
    prune_ratio: float = 0.4,
    strategy: str = "magnitude_random",
    seed: int = 42,
) -> dict[str, torch.Tensor]:
    """
    Create binary masks for neuron-level pruning of GCN hidden layers.

    Strategies:
    - "random": Uniformly random selection of neurons to keep.
    - "magnitude": Keep neurons with highest L1-norm weights.
    - "magnitude_random": Hybrid — rank by magnitude, then randomly prune
      from the bottom 60% (preserves some diversity).

    Parameters
    ----------
    model : nn.Module
        Pre-trained baseline GCN model.
    prune_ratio : float
        Fraction of neurons to REMOVE (e.g., 0.4 = remove 40%).
    strategy : str
        Pruning strategy: "random", "magnitude", or "magnitude_random".

    Returns
    -------
    masks : dict mapping layer name → binary mask Tensor of shape (hidden_dim,).
    """
    rng = np.random.RandomState(seed)
    masks = {}

    for name, module in model.named_modules():
        if isinstance(module, GCNConv):
            # GCNConv has a linear transformation: module.lin.weight (out, in)
            weight = module.lin.weight.data  # (out_channels, in_channels)
            out_channels = weight.shape[0]
            num_keep = max(1, int(out_channels * (1.0 - prune_ratio)))

            if strategy == "random":
                keep_idx = rng.choice(out_channels, size=num_keep, replace=False)
                mask = torch.zeros(out_channels, device=weight.device)
                mask[keep_idx] = 1.0

            elif strategy == "magnitude":
                # L1 norm per output neuron
                importance = weight.abs().sum(dim=1).cpu().numpy()
                keep_idx = np.argsort(importance)[-num_keep:]
                mask = torch.zeros(out_channels, device=weight.device)
                mask[keep_idx] = 1.0

            elif strategy == "magnitude_random":
                # Rank by magnitude, randomly prune from bottom 60%
                importance = weight.abs().sum(dim=1).cpu().numpy()
                ranked = np.argsort(importance)
                # Always keep top 40% by magnitude
                top_keep = max(1, int(out_channels * 0.4))
                must_keep = set(ranked[-top_keep:].tolist())
                # Randomly select from remaining to fill quota
                remaining = [i for i in range(out_channels) if i not in must_keep]
                extra_needed = num_keep - len(must_keep)
                if extra_needed > 0 and len(remaining) > 0:
                    extra = rng.choice(remaining, size=min(extra_needed, len(remaining)),
                                       replace=False)
                    must_keep.update(extra.tolist())
                mask = torch.zeros(out_channels, device=weight.device)
                for idx in must_keep:
                    mask[idx] = 1.0
            else:
                raise ValueError(f"Unknown pruning strategy: {strategy}")

            masks[name] = mask

    return masks


def apply_pruning_masks(model: nn.Module, masks: dict[str, torch.Tensor]) -> nn.Module:
    """
    Apply neuron pruning masks to a GCN model in-place.

    Zeroes out the weights of pruned neurons. During fine-tuning, gradients
    for pruned neurons are also zeroed via hooks.

    Parameters
    ----------
    model : nn.Module
        The model to prune.
    masks : dict
        Neuron masks from create_pruning_masks().

    Returns
    -------
    model : nn.Module — same model with masks applied.
    """
    hooks = []

    for name, module in model.named_modules():
        if name in masks:
            mask = masks[name]

            # Zero out pruned neuron weights
            with torch.no_grad():
                # GCNConv linear weight: (out, in)
                module.lin.weight.data *= mask.unsqueeze(1)
                if module.lin.bias is not None:
                    module.lin.bias.data *= mask

            # Register hook to zero gradients of pruned neurons during training
            def _make_hook(m, msk):
                def hook_fn(grad):
                    return grad * msk.unsqueeze(1)
                m.lin.weight.register_hook(hook_fn)
                if m.lin.bias is not None:
                    def bias_hook_fn(grad):
                        return grad * msk
                    m.lin.bias.register_hook(bias_hook_fn)
            _make_hook(module, mask)

    # Also apply masks to corresponding BatchNorm layers
    bn_idx = 0
    for name, module in model.named_modules():
        if isinstance(module, nn.BatchNorm1d):
            # Find the corresponding conv layer mask
            conv_names = [n for n in masks.keys()]
            if bn_idx < len(conv_names):
                mask = masks[conv_names[bn_idx]]
                with torch.no_grad():
                    module.weight.data *= mask
                    module.bias.data *= mask
                    module.running_mean *= mask
                    module.running_var = module.running_var * mask + (1 - mask)
                bn_idx += 1

    return model


# ---------------------------------------------------------------------------
# SLIDE Fine-tuning Loop
# ---------------------------------------------------------------------------

def slide_finetune(
    model: nn.Module,
    train_loader,
    val_loader,
    masks: dict[str, torch.Tensor] | None = None,
    epochs: int = 50,
    lr: float = 5e-4,
    weight_decay: float = 1e-4,
    lambda_decorr: float = 0.1,
    rff_dim: int = 256,
    rff_sigma: float = 1.0,
    patience: int = 10,
    device: str = "cuda",
    save_path: str = "checkpoints/slide_pruned.pt",
):
    """
    Fine-tune the pruned model with SLIDE de-correlation loss.

    Loss = CrossEntropy + λ * L_decorr

    The de-correlation penalty forces surviving neurons to learn diverse,
    non-redundant representations, compensating for the capacity lost
    during pruning.

    Parameters
    ----------
    model : nn.Module
        Pruned baseline GCN model.
    train_loader, val_loader : DataLoader
        Training and validation data.
    masks : dict
        Pruning masks (for gradient zeroing).
    epochs : int
        Maximum fine-tuning epochs.
    lr : float
        Learning rate.
    lambda_decorr : float
        Weight of de-correlation penalty.
    rff_dim : int
        Number of Random Fourier Features.
    rff_sigma : float
        RBF kernel bandwidth for RFF.
    patience : int
        Early stopping patience.

    Returns
    -------
    model : nn.Module — fine-tuned pruned model.
    history : dict
    """
    import os
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    model = model.to(device)

    # Determine hidden dim from last conv layer (output dim of node embeddings)
    hidden_dim = None
    for name, module in model.named_modules():
        if isinstance(module, GCNConv):
            hidden_dim = module.lin.weight.shape[0]
    assert hidden_dim is not None, "No GCNConv layer found in model"

    # Initialize RFF module
    rff = RandomFourierFeatures(in_dim=hidden_dim, rff_dim=rff_dim, sigma=rff_sigma).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    criterion = nn.CrossEntropyLoss()

    best_val_loss = float("inf")
    patience_counter = 0
    history = {"train_loss": [], "train_ce": [], "train_decorr": [],
               "val_loss": [], "val_acc": []}

    for epoch in range(1, epochs + 1):
        # --- Train ---
        model.train()
        total_loss = 0
        total_ce = 0
        total_decorr = 0
        num_samples = 0

        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()

            # Forward: get logits
            out = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
            ce_loss = criterion(out, batch.y.view(-1))

            # Get node embeddings for de-correlation penalty
            node_emb = model.get_node_embeddings(
                batch.x, batch.edge_index, batch.edge_attr
            )
            decorr = decorrelation_loss(node_emb, rff)

            loss = ce_loss + lambda_decorr * decorr
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            # Re-apply masks after optimizer step (ensure pruned weights stay zero)
            if masks:
                with torch.no_grad():
                    for name, module in model.named_modules():
                        if name in masks:
                            mask = masks[name].to(device)
                            module.lin.weight.data *= mask.unsqueeze(1)
                            if module.lin.bias is not None:
                                module.lin.bias.data *= mask

            total_loss += loss.item() * batch.num_graphs
            total_ce += ce_loss.item() * batch.num_graphs
            total_decorr += decorr.item() * batch.num_graphs
            num_samples += batch.num_graphs

        train_loss = total_loss / num_samples
        train_ce = total_ce / num_samples
        train_decorr = total_decorr / num_samples
        scheduler.step()

        # --- Validate ---
        model.eval()
        val_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                out = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
                loss = criterion(out, batch.y.view(-1))
                val_loss += loss.item() * batch.num_graphs
                pred = out.argmax(dim=1)
                correct += (pred == batch.y.view(-1)).sum().item()
                total += batch.num_graphs

        val_loss /= total
        val_acc = correct / total

        history["train_loss"].append(train_loss)
        history["train_ce"].append(train_ce)
        history["train_decorr"].append(train_decorr)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        if epoch % 10 == 0 or epoch == 1:
            print(f"  [SLIDE] Epoch {epoch:3d} | Loss: {train_loss:.4f} "
                  f"(CE: {train_ce:.4f} + λ·Decorr: {lambda_decorr * train_decorr:.4f}) | "
                  f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")

        # Early stopping
        if val_loss < best_val_loss - 1e-4:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save({
                "model_state_dict": model.state_dict(),
                "masks": masks if masks else {},
            }, save_path)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"  [SLIDE] Early stopping at epoch {epoch}")
                break

    # Load best model
    ckpt = torch.load(save_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    print(f"[SLIDE] Best val_loss: {best_val_loss:.4f}")
    return model, history


# ---------------------------------------------------------------------------
# Model compaction: physically shrink the network
# ---------------------------------------------------------------------------

def compact_model(
    model: nn.Module,
    masks: dict[str, torch.Tensor],
) -> nn.Module:
    """
    Create a physically smaller model by removing pruned neurons entirely.

    Instead of keeping full-size tensors with zeros, this builds a new
    BaselineGCN with reduced hidden_channels and copies only the surviving
    weights. The result is a genuinely smaller model with lower memory
    footprint and faster inference.

    Parameters
    ----------
    model : nn.Module
        Mask-pruned BaselineGCN (from apply_pruning_masks).
    masks : dict
        Pruning masks from create_pruning_masks().

    Returns
    -------
    compact : BaselineGCN
        Physically smaller model with reduced layer widths.
    """
    from baseline_model import BaselineGCN

    # Determine kept neuron indices per GCN layer
    conv_names = sorted(masks.keys())  # e.g., ['convs.0', 'convs.1', 'convs.2']
    keep_indices = []
    hidden_sizes = []
    for cname in conv_names:
        mask = masks[cname]
        keep_idx = torch.where(mask > 0.5)[0]
        keep_indices.append(keep_idx)
        hidden_sizes.append(len(keep_idx))

    # Use the minimum hidden size for the compact model
    # (BaselineGCN uses uniform hidden_channels across all layers)
    compact_hidden = min(hidden_sizes)
    in_channels = model.convs[0].in_channels
    num_classes = model.mlp[-1].out_features
    num_gcn_layers = len(conv_names)

    compact = BaselineGCN(
        in_channels=in_channels,
        hidden_channels=compact_hidden,
        num_classes=num_classes,
        num_gcn_layers=num_gcn_layers,
        dropout=model.dropout,
    )

    # Transfer surviving weights
    with torch.no_grad():
        for i, cname in enumerate(conv_names):
            src_conv = model.convs[i]
            dst_conv = compact.convs[i]
            src_bn = model.bns[i]
            dst_bn = compact.bns[i]

            keep = keep_indices[i][:compact_hidden]  # trim to compact_hidden

            if i == 0:
                # First layer: select output neurons, keep all input channels
                dst_conv.lin.weight.copy_(src_conv.lin.weight.data[keep, :])
                if src_conv.lin.bias is not None and dst_conv.lin.bias is not None:
                    dst_conv.lin.bias.copy_(src_conv.lin.bias.data[keep])
            else:
                prev_keep = keep_indices[i - 1][:compact_hidden]
                # Select both input (from prev layer survivors) and output neurons
                w = src_conv.lin.weight.data[keep, :][:, prev_keep]
                dst_conv.lin.weight.copy_(w)
                if src_conv.lin.bias is not None and dst_conv.lin.bias is not None:
                    dst_conv.lin.bias.copy_(src_conv.lin.bias.data[keep])

            # BatchNorm
            dst_bn.weight.copy_(src_bn.weight.data[keep])
            dst_bn.bias.copy_(src_bn.bias.data[keep])
            dst_bn.running_mean.copy_(src_bn.running_mean.data[keep])
            dst_bn.running_var.copy_(src_bn.running_var.data[keep])

        # MLP head
        # Pool output = 2 * compact_hidden (mean || max of last GCN layer)
        last_keep = keep_indices[-1][:compact_hidden]
        orig_hidden = model.convs[0].out_channels

        # Pool indices: [kept neurons from mean pool, kept neurons from max pool]
        pool_keep = torch.cat([last_keep, last_keep + orig_hidden])

        # MLP layer 0: Linear(2*hidden → hidden)
        src_w0 = model.mlp[0].weight.data  # (hidden_orig, 2*hidden_orig)
        src_b0 = model.mlp[0].bias.data
        dst_w0 = src_w0[last_keep, :][:, pool_keep]
        compact.mlp[0].weight.copy_(dst_w0)
        compact.mlp[0].bias.copy_(src_b0[last_keep])

        # MLP layer 3: Linear(hidden → num_classes)
        src_w1 = model.mlp[3].weight.data  # (num_classes, hidden_orig)
        src_b1 = model.mlp[3].bias.data
        compact.mlp[3].weight.copy_(src_w1[:, last_keep])
        compact.mlp[3].bias.copy_(src_b1)

    print(f"[Compact] {orig_hidden} → {compact_hidden} hidden channels | "
          f"Parameters: {compact.count_parameters():,}")
    return compact


# ---------------------------------------------------------------------------
# Compute active (non-pruned) parameter count
# ---------------------------------------------------------------------------

def count_active_parameters(model: nn.Module, masks: dict[str, torch.Tensor]) -> int:
    """
    Count trainable parameters excluding pruned neurons.

    For pruned GCN layers, only counts rows (neurons) where mask == 1.
    For other layers, counts all trainable parameters.
    """
    total = 0

    pruned_layers = set(masks.keys())

    for name, module in model.named_modules():
        if name in pruned_layers:
            mask = masks[name]
            active_neurons = int(mask.sum().item())
            in_features = module.lin.weight.shape[1]
            # Weight: active_neurons * in_features
            total += active_neurons * in_features
            # Bias
            if module.lin.bias is not None:
                total += active_neurons
        elif hasattr(module, 'weight') and module.weight is not None:
            # Skip if this is a sub-module of a named module we already counted
            if not any(name.startswith(pn) for pn in pruned_layers):
                if module.weight.requires_grad:
                    total += module.weight.numel()
                if hasattr(module, 'bias') and module.bias is not None and module.bias.requires_grad:
                    total += module.bias.numel()

    return total
