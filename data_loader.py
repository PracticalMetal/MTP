"""
Data Loading Module for IoT Intrusion Detection
=================================================
Loads TON_IoT dataset or generates simulated IoT network traffic graphs.
Each graph: nodes = IoT devices, edges = communication links weighted by
frequency (f_ij) and data volume (d_ij): w_ij = alpha * f_ij + beta * d_ij.

Node features encode device-level traffic statistics + protocol one-hot.
Graph label: 0 = Normal, 1 = Attack (DoS, Spoofing, MITM).

Reference: Booij et al., "TON_IoT: The role of heterogeneity and the need
for standardization of features and attack types in IoT network IDS datasets,"
IEEE Internet of Things Journal, 2022.
"""

import os
import random
import numpy as np
import torch
from torch_geometric.data import Data, InMemoryDataset, Dataset
from torch_geometric.utils import to_undirected
from sklearn.preprocessing import StandardScaler


# ---------------------------------------------------------------------------
# Protocol encoding (one-hot index)
# ---------------------------------------------------------------------------
PROTOCOLS = ["MQTT", "CoAP", "Zigbee", "TCP", "UDP"]
PROTO2IDX = {p: i for i, p in enumerate(PROTOCOLS)}
NUM_PROTOCOLS = len(PROTOCOLS)

# Attack types for simulation
ATTACK_TYPES = ["Normal", "DoS", "Spoofing", "MITM"]


# ---------------------------------------------------------------------------
# Helper: build a single IoT traffic graph
# ---------------------------------------------------------------------------
def _build_iot_graph(
    num_nodes: int,
    edge_density: float,
    attack: bool,
    alpha: float = 0.6,
    beta: float = 0.4,
    feature_dim: int = 16,
    seed: int | None = None,
) -> Data:
    """
    Construct one IoT traffic graph.

    Parameters
    ----------
    num_nodes : int
        Number of IoT devices (graph nodes).
    edge_density : float
        Fraction of possible edges that are present (sparse for IoT).
    attack : bool
        Whether the graph represents an attack scenario.
    alpha, beta : float
        Weights for edge weight formula: w = alpha*freq + beta*data_vol.
    feature_dim : int
        Dimension of per-node traffic feature vector (excluding protocol one-hot).
    seed : int or None
        Random seed for reproducibility.
    """
    rng = np.random.RandomState(seed)

    # --- Node features ---
    # Traffic statistics: packet rate, byte rate, flow duration, etc.
    if attack:
        # Attacks: same base distribution as normal. Per-graph random intensity
        # controls how detectable the attack is — some attacks are stealthy,
        # some are obvious. This creates a realistic difficulty gradient.
        x_traffic = rng.normal(loc=1.0, scale=0.5, size=(num_nodes, feature_dim)).astype(np.float32)
        x_traffic = np.clip(x_traffic, 0.01, None)
        # Random attack intensity per graph: low (0.5) = stealthy, high (3.0) = obvious
        intensity = rng.uniform(0.5, 3.0)
        n_sources = max(1, int(num_nodes * rng.uniform(0.08, 0.30)))
        source_ids = rng.choice(num_nodes, n_sources, replace=False)
        n_affected_feats = max(1, int(feature_dim * rng.uniform(0.2, 0.6)))
        affected_feats = rng.choice(feature_dim, n_affected_feats, replace=False)
        x_traffic[np.ix_(source_ids, affected_feats)] += rng.exponential(
            scale=intensity, size=(n_sources, n_affected_feats)
        ).astype(np.float32)
    else:
        # Normal: some graphs also have natural traffic spikes (false-positive bait)
        x_traffic = rng.normal(loc=1.0, scale=0.5, size=(num_nodes, feature_dim)).astype(np.float32)
        x_traffic = np.clip(x_traffic, 0.01, None)
        # ~10% of normal graphs have benign traffic spikes (e.g., firmware updates)
        if rng.random() < 0.15:
            n_spike = max(1, num_nodes // 10)
            spike_ids = rng.choice(num_nodes, n_spike, replace=False)
            n_spike_feats = max(1, feature_dim // 4)
            spike_feats = rng.choice(feature_dim, n_spike_feats, replace=False)
            x_traffic[np.ix_(spike_ids, spike_feats)] += rng.exponential(
                scale=0.8, size=(n_spike, n_spike_feats)
            ).astype(np.float32)

    # Protocol assignment per node
    proto_indices = rng.choice(NUM_PROTOCOLS, size=num_nodes)
    x_proto = np.zeros((num_nodes, NUM_PROTOCOLS), dtype=np.float32)
    x_proto[np.arange(num_nodes), proto_indices] = 1.0

    # Full node feature: [traffic_stats | protocol_one_hot]
    x = np.concatenate([x_traffic, x_proto], axis=1)  # (N, feature_dim + NUM_PROTOCOLS)

    # --- Edges ---
    max_edges = num_nodes * (num_nodes - 1) // 2
    num_edges = max(num_nodes - 1, int(max_edges * edge_density))  # at least a tree

    # Start with a random spanning tree for connectivity
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

    # Add random extra edges
    added = set()
    for s, d in zip(src_list[::2], dst_list[::2]):
        added.add((min(s, d), max(s, d)))

    extra_needed = num_edges - (num_nodes - 1)
    attempts = 0
    while len(added) < num_edges and attempts < extra_needed * 10:
        s = rng.randint(0, num_nodes)
        d = rng.randint(0, num_nodes)
        if s == d:
            attempts += 1
            continue
        edge_key = (min(s, d), max(s, d))
        if edge_key not in added:
            added.add(edge_key)
            freq = rng.exponential(1.0)
            data_vol = rng.exponential(1.0)
            if attack:
                # Attack edges: subtle volume variation
                data_vol *= rng.uniform(1.0, 1.5)
            w = alpha * freq + beta * data_vol
            src_list.extend([s, d])
            dst_list.extend([d, s])
            weight_list.extend([w, w])
        attempts += 1

    edge_index = torch.tensor([src_list, dst_list], dtype=torch.long)
    edge_weight = torch.tensor(weight_list, dtype=torch.float32)

    # Normalize features
    x = torch.tensor(x, dtype=torch.float32)

    # Graph-level label
    y = torch.tensor([1 if attack else 0], dtype=torch.long)

    return Data(x=x, edge_index=edge_index, edge_attr=edge_weight, y=y,
                num_nodes=num_nodes)


# ---------------------------------------------------------------------------
# Simulated IoT Dataset
# ---------------------------------------------------------------------------
class SimulatedIoTDataset(InMemoryDataset):
    """
    Generates a synthetic IoT network traffic dataset for development/testing.

    Parameters
    ----------
    root : str
        Directory to store processed data.
    num_graphs : int
        Total number of graphs.
    min_nodes, max_nodes : int
        Range for number of devices per graph.
    edge_density : float
        Fraction of possible edges present.
    attack_ratio : float
        Fraction of graphs that are attack scenarios.
    feature_dim : int
        Dimension of traffic feature vector per node.
    seed : int
        Base random seed.
    """

    def __init__(
        self,
        root: str = "./data/simulated",
        num_graphs: int = 2000,
        min_nodes: int = 20,
        max_nodes: int = 200,
        edge_density: float = 0.05,
        attack_ratio: float = 0.4,
        feature_dim: int = 16,
        seed: int = 42,
        transform=None,
        pre_transform=None,
    ):
        self.num_graphs = num_graphs
        self.min_nodes = min_nodes
        self.max_nodes = max_nodes
        self.edge_density = edge_density
        self.attack_ratio = attack_ratio
        self.feature_dim = feature_dim
        self.seed = seed
        super().__init__(root, transform, pre_transform)
        self.load(self.processed_paths[0])

    @property
    def processed_file_names(self):
        return [f"simulated_iot_{self.num_graphs}_{self.seed}.pt"]

    def process(self):
        rng = np.random.RandomState(self.seed)
        data_list = []
        num_attacks = int(self.num_graphs * self.attack_ratio)
        labels = [1] * num_attacks + [0] * (self.num_graphs - num_attacks)
        rng.shuffle(labels)

        for i, label in enumerate(labels):
            n = rng.randint(self.min_nodes, self.max_nodes + 1)
            g = _build_iot_graph(
                num_nodes=n,
                edge_density=self.edge_density,
                attack=bool(label),
                feature_dim=self.feature_dim,
                seed=self.seed + i,
            )
            data_list.append(g)

        if self.pre_transform is not None:
            data_list = [self.pre_transform(d) for d in data_list]

        self.save(data_list, self.processed_paths[0])


# ---------------------------------------------------------------------------
# TON_IoT Dataset Loader (CSV-based)
# ---------------------------------------------------------------------------
class TONIoTDataset(InMemoryDataset):
    """
    Loads the TON_IoT dataset from CSV files and constructs per-window graphs.

    Expects CSV files in `root/raw/` with columns including:
      src_ip, dst_ip, protocol, ts, flow_duration, fwd_pkts, bwd_pkts,
      fwd_bytes, bwd_bytes, ..., label (0=Normal, 1=Attack), attack_type

    The loader groups packets into time windows and builds one graph per window
    where nodes = unique IPs, edges = communication pairs.

    If CSV files are not found, falls back to simulated data with a warning.

    Parameters
    ----------
    root : str
        Directory containing raw/ subfolder with CSVs.
    window_size : float
        Time window in seconds for graph construction.
    feature_dim : int
        Node feature dimension (traffic stats).
    """

    def __init__(
        self,
        root: str = "./data/ton_iot",
        window_size: float = 10.0,
        feature_dim: int = 16,
        transform=None,
        pre_transform=None,
    ):
        self.window_size = window_size
        self.feature_dim = feature_dim
        super().__init__(root, transform, pre_transform)
        self.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return ["network_traffic.csv"]

    @property
    def processed_file_names(self):
        return ["ton_iot_graphs.pt"]

    def download(self):
        pass  # User must place CSVs manually

    def process(self):
        import csv
        raw_path = os.path.join(self.raw_dir, "network_traffic.csv")

        if not os.path.exists(raw_path):
            print(f"[WARNING] TON_IoT CSV not found at {raw_path}.")
            print("[WARNING] Generating simulated fallback data (2000 graphs).")
            self._process_simulated_fallback()
            return

        self._process_csv(raw_path)

    def _process_csv(self, csv_path: str):
        """Parse real TON_IoT CSV and construct time-windowed graphs."""
        import pandas as pd

        df = pd.read_csv(csv_path)
        required = ["src_ip", "dst_ip", "protocol", "ts", "label"]
        for col in required:
            assert col in df.columns, f"Missing column: {col}"

        df = df.sort_values("ts").reset_index(drop=True)

        # Discretize into time windows
        t_min = df["ts"].min()
        df["window"] = ((df["ts"] - t_min) / self.window_size).astype(int)

        data_list = []
        scaler = StandardScaler()

        for wid, group in df.groupby("window"):
            if len(group) < 5:
                continue

            # Build node mapping
            ips = list(set(group["src_ip"].tolist() + group["dst_ip"].tolist()))
            ip2idx = {ip: i for i, ip in enumerate(ips)}
            num_nodes = len(ips)

            # Node features: aggregate traffic stats per IP
            x = np.zeros((num_nodes, self.feature_dim + NUM_PROTOCOLS), dtype=np.float32)
            for _, row in group.iterrows():
                si = ip2idx[row["src_ip"]]
                di = ip2idx[row["dst_ip"]]
                # Increment packet/byte counters (first feature_dim dims)
                feat_vals = []
                for col in ["fwd_pkts", "bwd_pkts", "fwd_bytes", "bwd_bytes",
                            "flow_duration"]:
                    feat_vals.append(float(row.get(col, 0.0)))
                # Pad to feature_dim
                feat_vals = feat_vals + [0.0] * (self.feature_dim - len(feat_vals))
                x[si, :self.feature_dim] += np.array(feat_vals[:self.feature_dim])
                x[di, :self.feature_dim] += np.array(feat_vals[:self.feature_dim]) * 0.5

                # Protocol one-hot
                proto = row.get("protocol", "TCP")
                pidx = PROTO2IDX.get(str(proto).upper(), PROTO2IDX["TCP"])
                x[si, self.feature_dim + pidx] = 1.0
                x[di, self.feature_dim + pidx] = 1.0

            # Edges
            edge_pairs = {}
            for _, row in group.iterrows():
                si = ip2idx[row["src_ip"]]
                di = ip2idx[row["dst_ip"]]
                key = (min(si, di), max(si, di))
                if key not in edge_pairs:
                    edge_pairs[key] = 0.0
                edge_pairs[key] += 1.0  # frequency-based weight

            src_l, dst_l, w_l = [], [], []
            for (s, d), w in edge_pairs.items():
                src_l.extend([s, d])
                dst_l.extend([d, s])
                w_l.extend([w, w])

            edge_index = torch.tensor([src_l, dst_l], dtype=torch.long)
            edge_weight = torch.tensor(w_l, dtype=torch.float32)

            # Label: attack if majority of flows in window are attacks
            label = int(group["label"].mean() > 0.5)
            y = torch.tensor([label], dtype=torch.long)

            data = Data(
                x=torch.tensor(x, dtype=torch.float32),
                edge_index=edge_index,
                edge_attr=edge_weight,
                y=y,
                num_nodes=num_nodes,
            )
            data_list.append(data)

        if self.pre_transform is not None:
            data_list = [self.pre_transform(d) for d in data_list]

        self.save(data_list, self.processed_paths[0])
        print(f"[TON_IoT] Processed {len(data_list)} graphs from CSV.")

    def _process_simulated_fallback(self):
        """Generate simulated data as fallback when CSVs aren't available."""
        rng = np.random.RandomState(42)
        data_list = []
        for i in range(2000):
            n = rng.randint(20, 201)
            attack = rng.random() < 0.4
            g = _build_iot_graph(
                num_nodes=n,
                edge_density=0.05,
                attack=attack,
                feature_dim=self.feature_dim,
                seed=42 + i,
            )
            data_list.append(g)

        if self.pre_transform is not None:
            data_list = [self.pre_transform(d) for d in data_list]

        self.save(data_list, self.processed_paths[0])
        print(f"[TON_IoT-Simulated] Generated {len(data_list)} fallback graphs.")


# ---------------------------------------------------------------------------
# Utility: get train/val/test DataLoaders
# ---------------------------------------------------------------------------
def get_dataloaders(
    dataset,
    batch_size: int = 64,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    seed: int = 42,
):
    """Split dataset and return PyG DataLoaders."""
    from torch_geometric.loader import DataLoader

    n = len(dataset)
    indices = list(range(n))
    rng = random.Random(seed)
    rng.shuffle(indices)

    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    train_idx = indices[:n_train]
    val_idx = indices[n_train : n_train + n_val]
    test_idx = indices[n_train + n_val :]

    train_loader = DataLoader([dataset[i] for i in train_idx], batch_size=batch_size, shuffle=True)
    val_loader = DataLoader([dataset[i] for i in val_idx], batch_size=batch_size, shuffle=False)
    test_loader = DataLoader([dataset[i] for i in test_idx], batch_size=batch_size, shuffle=False)

    print(f"[Data] Train: {len(train_idx)} | Val: {len(val_idx)} | Test: {len(test_idx)}")
    return train_loader, val_loader, test_loader
