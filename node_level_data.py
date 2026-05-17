"""
Node-Level IoT Traffic Graph Generator
=======================================
Generates a single large IoT network graph with node-level labels.
Nodes = IoT devices, Edges = communication links.
Label: 0 = Normal, 1 = Attack (per-node).

Design principles for realistic difficulty:
  - Node features alone are NOT sufficient for high-accuracy classification.
    Attack features have subtle, overlapping anomalies (not 5x amplification).
  - Graph structure carries essential signal: same-class nodes cluster together
    (homophily), providing GNN message-passing with useful neighborhood info.
  - Noisy cross-class edges exist that confuse GNN message passing.
    Pruning these noisy edges should genuinely improve classification.
"""

import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.utils import add_self_loops


PROTOCOLS = ["MQTT", "CoAP", "Zigbee", "TCP", "UDP"]
NUM_PROTOCOLS = len(PROTOCOLS)


def generate_iot_graph(
    num_nodes: int = 5000,
    edge_density: float = 0.005,
    attack_ratio: float = 0.3,
    feature_dim: int = 16,
    alpha: float = 0.6,
    beta: float = 0.4,
    noise_edge_ratio: float = 0.3,
    seed: int = 42,
    add_loops: bool = True,
) -> Data:
    """
    Generate a single IoT network graph with node-level anomaly labels.

    The graph has three types of edges:
      1. Intra-class edges (normal-normal, attack-attack) — useful homophily
      2. Structural edges (spanning tree) — ensure connectivity
      3. Noisy cross-class edges — confuse GNN, should be pruned

    Parameters
    ----------
    num_nodes : int
        Number of IoT devices (nodes).
    edge_density : float
        Fraction of possible edges present.
    attack_ratio : float
        Fraction of nodes that are attack nodes.
    feature_dim : int
        Dimension of per-node traffic features (excluding protocol one-hot).
    noise_edge_ratio : float
        Fraction of non-tree edges that are noisy cross-class edges.
    seed : int
        Random seed.
    """
    rng = np.random.RandomState(seed)

    # --- Assign node labels ---
    num_attack = int(num_nodes * attack_ratio)
    labels = np.zeros(num_nodes, dtype=np.int64)
    attack_ids = rng.choice(num_nodes, num_attack, replace=False)
    labels[attack_ids] = 1
    normal_ids = np.where(labels == 0)[0]

    # --- Node features ---
    # SUBTLE differences: both classes share overlapping distributions.
    # Attack nodes have a small shift (not 5x amplification).
    # This ensures feature-only classifiers get ~0.75-0.80 F1, not 0.95+.
    x_traffic = rng.normal(loc=0.0, scale=1.0, size=(num_nodes, feature_dim)).astype(np.float32)

    # Attack nodes: shift a SUBSET of features by a modest amount
    for aid in attack_ids:
        # Shift 20-40% of features by 0.5-1.5 std deviations
        n_affected = max(1, int(feature_dim * rng.uniform(0.2, 0.4)))
        affected_feats = rng.choice(feature_dim, n_affected, replace=False)
        shift = rng.uniform(0.5, 1.5)
        x_traffic[aid, affected_feats] += shift

    # ~10% of normal nodes get slight upward shift (false-positive bait)
    n_spike_nodes = max(1, int(len(normal_ids) * 0.10))
    spike_nodes = rng.choice(normal_ids, n_spike_nodes, replace=False)
    for sid in spike_nodes:
        n_spike_feats = max(1, feature_dim // 5)
        spike_feats = rng.choice(feature_dim, n_spike_feats, replace=False)
        x_traffic[sid, spike_feats] += rng.uniform(0.3, 0.8)

    # Protocol assignment (one-hot) — attack nodes prefer certain protocols
    proto_indices = np.zeros(num_nodes, dtype=int)
    for i in range(num_nodes):
        if labels[i] == 1:
            # Attack nodes: biased toward UDP/TCP (indices 3,4)
            proto_indices[i] = rng.choice(NUM_PROTOCOLS, p=[0.1, 0.1, 0.1, 0.35, 0.35])
        else:
            # Normal nodes: biased toward MQTT/CoAP/Zigbee (indices 0,1,2)
            proto_indices[i] = rng.choice(NUM_PROTOCOLS, p=[0.3, 0.25, 0.25, 0.1, 0.1])

    x_proto = np.zeros((num_nodes, NUM_PROTOCOLS), dtype=np.float32)
    x_proto[np.arange(num_nodes), proto_indices] = 1.0

    # Concatenate: [traffic_stats | protocol_one_hot]
    x = np.concatenate([x_traffic, x_proto], axis=1)

    # --- Edges ---
    max_edges = num_nodes * (num_nodes - 1) // 2
    target_edges = max(num_nodes - 1, int(max_edges * edge_density))
    target_edges = min(target_edges, num_nodes * 20)

    # Phase 1: Random spanning tree for connectivity
    perm = rng.permutation(num_nodes)
    src_list, dst_list, weight_list = [], [], []
    for i in range(1, num_nodes):
        s, d = int(perm[i - 1]), int(perm[i])
        freq = rng.exponential(1.0)
        data_vol = rng.exponential(1.0)
        w = alpha * freq + beta * data_vol
        src_list.extend([s, d])
        dst_list.extend([d, s])
        weight_list.extend([w, w])

    edge_set = set()
    for s, d in zip(src_list[::2], dst_list[::2]):
        edge_set.add((min(s, d), max(s, d)))

    # Phase 2: Homophilic intra-class edges + noisy cross-class edges
    extra_needed = target_edges - (num_nodes - 1)
    n_noise = int(extra_needed * noise_edge_ratio)
    n_homophilic = extra_needed - n_noise

    def _add_edge(s, d, edge_set, src_list, dst_list, weight_list, is_noisy=False):
        """Add a single undirected edge."""
        if s == d:
            return False
        key = (min(s, d), max(s, d))
        if key in edge_set:
            return False
        edge_set.add(key)
        freq = rng.exponential(1.0)
        data_vol = rng.exponential(1.0)
        if is_noisy:
            # Noisy cross-class edges have slightly unusual weight patterns
            data_vol *= rng.uniform(0.3, 0.8)
        elif labels[s] == 1 and labels[d] == 1:
            # Attack-attack edges have higher volume
            data_vol *= rng.uniform(1.5, 2.5)
        w = alpha * freq + beta * data_vol
        src_list.extend([s, d])
        dst_list.extend([d, s])
        weight_list.extend([w, w])
        return True

    # Add homophilic edges (same-class connections)
    added_homo = 0
    attempts = 0
    while added_homo < n_homophilic and attempts < n_homophilic * 15:
        # Pick same-class pair
        if rng.random() < 0.5 and len(attack_ids) > 1:
            s = int(rng.choice(attack_ids))
            d = int(rng.choice(attack_ids))
        else:
            s = int(rng.choice(normal_ids))
            d = int(rng.choice(normal_ids))
        if _add_edge(s, d, edge_set, src_list, dst_list, weight_list, is_noisy=False):
            added_homo += 1
        attempts += 1

    # Add noisy cross-class edges (normal↔attack — these confuse GNN)
    added_noise = 0
    attempts = 0
    while added_noise < n_noise and attempts < n_noise * 15:
        s = int(rng.choice(normal_ids))
        d = int(rng.choice(attack_ids))
        if _add_edge(s, d, edge_set, src_list, dst_list, weight_list, is_noisy=True):
            added_noise += 1
        attempts += 1

    edge_index = torch.tensor([src_list, dst_list], dtype=torch.long)
    edge_attr = torch.tensor(weight_list, dtype=torch.float32)

    # Track which edges are noisy (for analysis, not used in training)
    # Noisy edges are the last `added_noise * 2` entries (undirected, so *2)
    n_total_edges = len(weight_list)
    is_noisy_edge = torch.zeros(n_total_edges, dtype=torch.bool)
    if added_noise > 0:
        is_noisy_edge[-(added_noise * 2):] = True

    # Add self-loops
    if add_loops:
        n_before_loops = edge_index.size(1)
        edge_index, edge_attr = add_self_loops(
            edge_index, edge_attr, fill_value=1.0, num_nodes=num_nodes
        )
        # Extend noisy mask for self-loops (self-loops are not noisy)
        n_loops = edge_index.size(1) - n_before_loops
        is_noisy_edge = torch.cat([is_noisy_edge, torch.zeros(n_loops, dtype=torch.bool)])

    x = torch.tensor(x, dtype=torch.float32)
    y = torch.tensor(labels, dtype=torch.long)

    # --- Train/Val/Test splits ---
    indices = rng.permutation(num_nodes)
    n_train = int(num_nodes * 0.6)
    n_val = int(num_nodes * 0.2)

    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)

    train_mask[indices[:n_train]] = True
    val_mask[indices[n_train:n_train + n_val]] = True
    test_mask[indices[n_train + n_val:]] = True

    data = Data(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
        y=y,
        num_nodes=num_nodes,
        train_mask=train_mask,
        val_mask=val_mask,
        test_mask=test_mask,
    )
    # Store noise mask as auxiliary info (not used in training)
    data.is_noisy_edge = is_noisy_edge

    print(f"[Data] Nodes: {num_nodes} | Edges: {edge_index.size(1)} | "
          f"Attack: {num_attack} ({100*attack_ratio:.0f}%)")
    print(f"[Data] Homophilic edges: {added_homo} | Noisy cross-class: {added_noise} | "
          f"Tree: {num_nodes-1}")
    print(f"[Data] Train: {train_mask.sum().item()} | "
          f"Val: {val_mask.sum().item()} | "
          f"Test: {test_mask.sum().item()}")

    return data
