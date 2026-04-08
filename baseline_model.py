"""
Baseline GCN Model for IoT Intrusion Detection (Phase 1)
=========================================================
Architecture: GCN layers → Global Mean/Max Pooling → MLP → Binary classification.
Matches Phase 1 performance targets: F1 ~0.95, AUC-ROC ~0.98, <2.5ms inference.

Uses edge weights from dynamic graph construction:
  w_ij = alpha * f_ij + beta * d_ij  (communication frequency + data volume)

Reference: Kipf & Welling, "Semi-Supervised Classification with Graph
Convolutional Networks," ICLR 2017.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool, global_max_pool


class BaselineGCN(nn.Module):
    """
    Phase 1 Baseline GCN for graph-level binary classification.

    Architecture:
        Input → GCN1 → ReLU → Dropout → GCN2 → ReLU → Dropout → GCN3 → ReLU →
        GlobalPool (mean || max) → MLP(2*hidden → hidden → 2) → Softmax

    Parameters
    ----------
    in_channels : int
        Node feature dimension (feature_dim + NUM_PROTOCOLS).
    hidden_channels : int
        Hidden layer width for GCN layers.
    num_classes : int
        Number of output classes (2 for binary: Normal/Attack).
    num_gcn_layers : int
        Number of GCN layers (default: 3).
    dropout : float
        Dropout rate (default: 0.5 per Phase 1 spec).
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = 128,
        num_classes: int = 2,
        num_gcn_layers: int = 3,
        dropout: float = 0.5,
    ):
        super().__init__()
        self.dropout = dropout
        self.num_gcn_layers = num_gcn_layers

        # GCN layers
        self.convs = nn.ModuleList()
        self.convs.append(GCNConv(in_channels, hidden_channels))
        for _ in range(num_gcn_layers - 1):
            self.convs.append(GCNConv(hidden_channels, hidden_channels))

        # Batch norms for each GCN layer
        self.bns = nn.ModuleList([
            nn.BatchNorm1d(hidden_channels) for _ in range(num_gcn_layers)
        ])

        # MLP classifier head
        # Pool concatenates mean + max → 2 * hidden_channels
        self.mlp = nn.Sequential(
            nn.Linear(2 * hidden_channels, hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, num_classes),
        )

    def get_node_embeddings(self, x, edge_index, edge_weight=None):
        """
        Forward through GCN layers only (no pooling/classifier).
        Returns node-level embeddings of shape (num_nodes, hidden_channels).
        Used by SLIDE for de-correlation penalty computation.
        """
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index, edge_weight=edge_weight)
            x = self.bns[i](x)
            x = F.relu(x)
            if i < self.num_gcn_layers - 1:
                x = F.dropout(x, p=self.dropout, training=self.training)
        return x

    def forward(self, x, edge_index, edge_weight=None, batch=None):
        """
        Full forward pass: GCN layers → pooling → MLP → logits.

        Parameters
        ----------
        x : Tensor (num_nodes, in_channels)
        edge_index : Tensor (2, num_edges)
        edge_weight : Tensor (num_edges,) or None
        batch : Tensor (num_nodes,) — batch assignment for graph-level pooling
        """
        # Node embeddings through GCN backbone
        h = self.get_node_embeddings(x, edge_index, edge_weight)

        # Global pooling: concatenate mean + max
        if batch is None:
            batch = torch.zeros(h.size(0), dtype=torch.long, device=h.device)
        h_mean = global_mean_pool(h, batch)  # (num_graphs, hidden)
        h_max = global_max_pool(h, batch)    # (num_graphs, hidden)
        h_pool = torch.cat([h_mean, h_max], dim=1)  # (num_graphs, 2*hidden)

        # MLP classification
        out = self.mlp(h_pool)  # (num_graphs, num_classes)
        return out

    def count_parameters(self) -> int:
        """Return total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def train_baseline(
    model: BaselineGCN,
    train_loader,
    val_loader,
    epochs: int = 100,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    patience: int = 15,
    device: str = "cuda",
    save_path: str = "checkpoints/baseline_gcn.pt",
):
    """
    Train the baseline GCN with early stopping on validation loss.

    Returns
    -------
    model : BaselineGCN
        Best model (loaded from checkpoint).
    history : dict
        Training history with 'train_loss', 'val_loss', 'val_acc' per epoch.
    """
    import os
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5, min_lr=1e-6
    )
    criterion = nn.CrossEntropyLoss()

    best_val_loss = float("inf")
    patience_counter = 0
    history = {"train_loss": [], "val_loss": [], "val_acc": []}

    for epoch in range(1, epochs + 1):
        # --- Train ---
        model.train()
        total_loss = 0
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            out = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
            loss = criterion(out, batch.y.view(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item() * batch.num_graphs

        train_loss = total_loss / len(train_loader.dataset)

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

        val_loss /= len(val_loader.dataset)
        val_acc = correct / total
        scheduler.step(val_loss)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        if epoch % 10 == 0 or epoch == 1:
            lr_now = optimizer.param_groups[0]["lr"]
            print(f"  Epoch {epoch:3d} | Train Loss: {train_loss:.4f} | "
                  f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | LR: {lr_now:.2e}")

        # Early stopping
        if val_loss < best_val_loss - 1e-4:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), save_path)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"  Early stopping at epoch {epoch} (best val_loss: {best_val_loss:.4f})")
                break

    # Load best model
    model.load_state_dict(torch.load(save_path, map_location=device, weights_only=True))
    print(f"[Baseline] Best val_loss: {best_val_loss:.4f} | Parameters: {model.count_parameters():,}")
    return model, history
