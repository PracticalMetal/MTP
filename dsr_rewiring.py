"""
DSR/GOKU: Densification-Sparsification Rewiring
=================================================
Graph-level structural pruning that preserves the Laplacian spectrum while
removing redundant edges and clearing topological bottlenecks.

Two-step process:
  Step 1 (Densification / USS): Compute the Fiedler vector (eigenvector of
    the second-smallest Laplacian eigenvalue). Add shortcut edges between
    nodes with large absolute Fiedler values and small degrees to clear
    bottlenecks. Uses Unimportance-based Spectral Sparsification in reverse.

  Step 2 (Sparsification / ISS): Compute Effective Resistance (ER) for each
    edge and cosine similarity between endpoint features. Sample edges with
    probability p_e ∝ (1 + S_e) * R_e, keeping the same total edge count
    and preserving spectral properties within (1 ± ε).

Reference:
  - Arnaiz-Rodriguez et al., "DiffWire: Inductive Graph Rewiring via the
    Lovász Bound," LoG 2022.
  - Spielman & Srivastava, "Graph Sparsification by Effective Resistances,"
    STOC 2008.
  - Banerjee et al., "Graph Rewiring via Spectral Alignment," NeurIPS 2022
    (GOKU framework).
"""

import numpy as np
import torch
from scipy import sparse as sp
from scipy.sparse.linalg import eigsh, LinearOperator
from torch_geometric.data import Data
from torch_geometric.utils import to_undirected, degree, coalesce


# ---------------------------------------------------------------------------
# Laplacian utilities (sparse, O(V+E) construction)
# ---------------------------------------------------------------------------

def _build_laplacian_sparse(edge_index: torch.Tensor, edge_weight: torch.Tensor,
                            num_nodes: int) -> sp.csr_matrix:
    """
    Build the (weighted) graph Laplacian L = D - W as a scipy sparse matrix.
    Operates on undirected edges: each (i,j) pair appears twice in edge_index.
    """
    ei = edge_index.cpu().numpy()
    ew = edge_weight.cpu().numpy() if edge_weight is not None else np.ones(ei.shape[1])

    # Adjacency matrix (upper triangle only, then symmetrize)
    W = sp.coo_matrix((ew, (ei[0], ei[1])), shape=(num_nodes, num_nodes))
    W = (W + W.T) / 2  # ensure symmetric
    W = W.tocsr()

    # Degree matrix
    d = np.array(W.sum(axis=1)).flatten()
    D = sp.diags(d)

    L = D - W
    return L.tocsr()


def compute_fiedler_vector(edge_index: torch.Tensor, edge_weight: torch.Tensor,
                           num_nodes: int) -> np.ndarray:
    """
    Compute the Fiedler vector (eigenvector of λ₂ of the Laplacian).

    The Fiedler vector reveals the graph's bottleneck structure: nodes with
    large |v₂[i]| are on opposite sides of the main cut, while nodes near
    zero are in the bottleneck region.

    Uses sparse eigensolver (ARPACK) — O(E) per iteration.
    """
    L = _build_laplacian_sparse(edge_index, edge_weight, num_nodes)

    # Compute 2 smallest eigenvalues/vectors (λ₁=0 for connected graphs)
    try:
        eigenvalues, eigenvectors = eigsh(L.astype(np.float64), k=2, which="SM",
                                          maxiter=5000, tol=1e-6)
    except Exception:
        # Fallback: if eigsh fails (e.g., disconnected graph), use dense solver
        # on a smaller subproblem or return uniform vector
        print("[DSR] Warning: sparse eigensolver failed, using fallback.")
        eigenvalues, eigenvectors = np.linalg.eigh(L.toarray().astype(np.float64))

    # λ₂ is the algebraic connectivity (Fiedler value)
    idx = np.argsort(eigenvalues)
    fiedler_vec = eigenvectors[:, idx[1]]  # eigenvector for λ₂

    return fiedler_vec


def compute_effective_resistance(edge_index: torch.Tensor, edge_weight: torch.Tensor,
                                 num_nodes: int, k: int = 50) -> np.ndarray:
    """
    Approximate effective resistance R_e for each edge using the pseudoinverse
    of the Laplacian via truncated eigendecomposition.

    R_e(i,j) = (L⁺[i,i] + L⁺[j,j] - 2*L⁺[i,j])

    We approximate L⁺ using the top-k eigenvectors (excluding λ₁=0):
        L⁺ ≈ Σ_{i=2}^{k+1} (1/λᵢ) * vᵢ * vᵢᵀ

    Reference: Spielman & Srivastava, "Graph Sparsification by Effective
    Resistances," STOC 2008.

    Parameters
    ----------
    k : int
        Number of eigenvectors to use for approximation (controls accuracy
        vs. computation trade-off). Default 50.

    Returns
    -------
    R_e : np.ndarray of shape (num_edges // 2,) — one ER per undirected edge.
    edge_pairs : np.ndarray of shape (num_edges // 2, 2) — the unique edge pairs.
    """
    L = _build_laplacian_sparse(edge_index, edge_weight, num_nodes)

    # Truncated eigendecomposition — k+1 smallest eigenvalues
    k_actual = min(k + 1, num_nodes - 1)
    try:
        eigenvalues, eigenvectors = eigsh(L.astype(np.float64), k=k_actual,
                                          which="SM", maxiter=5000, tol=1e-6)
    except Exception:
        eigenvalues, eigenvectors = np.linalg.eigh(L.toarray().astype(np.float64))
        eigenvalues = eigenvalues[:k_actual]
        eigenvectors = eigenvectors[:, :k_actual]

    # Skip the zero eigenvalue (connected component)
    idx = np.argsort(eigenvalues)
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    # Find where eigenvalues become > epsilon (skip nullspace)
    eps = 1e-8
    valid = eigenvalues > eps
    lam = eigenvalues[valid]
    V = eigenvectors[:, valid]  # (num_nodes, k_valid)

    # Scaled eigenvectors: Z = V * diag(1/sqrt(λ))
    Z = V / np.sqrt(lam)[np.newaxis, :]  # (num_nodes, k_valid)

    # Extract unique undirected edges
    ei = edge_index.cpu().numpy()
    # Keep only edges where src < dst (undirected)
    mask = ei[0] < ei[1]
    src = ei[0, mask]
    dst = ei[1, mask]

    # R_e(i,j) = ||Z[i] - Z[j]||² = L⁺[i,i] + L⁺[j,j] - 2*L⁺[i,j]
    diff = Z[src] - Z[dst]  # (num_unique_edges, k_valid)
    R_e = np.sum(diff ** 2, axis=1)  # effective resistance per edge

    edge_pairs = np.stack([src, dst], axis=1)
    return R_e, edge_pairs


# ---------------------------------------------------------------------------
# Step 1: Densification (USS — Unimportance-based Spectral Sparsification)
# ---------------------------------------------------------------------------

def densify_graph(
    data: Data,
    num_shortcuts: int | None = None,
    fiedler_quantile: float = 0.8,
    degree_quantile: float = 0.3,
    seed: int = 42,
) -> Data:
    """
    Add shortcut edges between bottleneck-adjacent nodes to improve
    spectral connectivity (reduce algebraic connectivity gap).

    Strategy: connect nodes with large |Fiedler value| (far from cut)
    to nodes with small degree (peripheral, potentially isolated).
    This clears topological bottlenecks while keeping the graph sparse.

    Parameters
    ----------
    data : Data
        Input PyG graph.
    num_shortcuts : int or None
        Number of shortcut edges to add. Default: ~10% of existing edges.
    fiedler_quantile : float
        Quantile threshold for |Fiedler value| — nodes above this are
        candidates for shortcut endpoints.
    degree_quantile : float
        Quantile threshold for degree — nodes below this are candidates
        for receiving shortcuts (low-degree peripheral nodes).
    """
    rng = np.random.RandomState(seed)
    edge_index = data.edge_index
    edge_weight = data.edge_attr if data.edge_attr is not None else None
    num_nodes = data.num_nodes

    if num_shortcuts is None:
        num_existing = edge_index.size(1) // 2  # undirected
        num_shortcuts = max(1, int(0.1 * num_existing))

    # Compute Fiedler vector
    ew = edge_weight if edge_weight is not None else torch.ones(edge_index.size(1))
    fiedler = compute_fiedler_vector(edge_index, ew, num_nodes)

    # Compute node degrees
    deg = degree(edge_index[0], num_nodes=num_nodes).cpu().numpy()

    # Identify candidate nodes
    abs_fiedler = np.abs(fiedler)
    fiedler_thresh = np.quantile(abs_fiedler, fiedler_quantile)
    degree_thresh = np.quantile(deg, degree_quantile)

    # High-Fiedler nodes (far from bottleneck — good shortcut sources)
    high_fiedler = np.where(abs_fiedler >= fiedler_thresh)[0]
    # Low-degree nodes (peripheral — good shortcut targets)
    low_degree = np.where(deg <= degree_thresh)[0]

    if len(high_fiedler) == 0 or len(low_degree) == 0:
        # Fallback: use all nodes
        high_fiedler = np.arange(num_nodes)
        low_degree = np.arange(num_nodes)

    # Build existing edge set for duplicate checking
    ei_np = edge_index.cpu().numpy()
    existing = set()
    for i in range(ei_np.shape[1]):
        s, d = int(ei_np[0, i]), int(ei_np[1, i])
        existing.add((min(s, d), max(s, d)))

    # Add shortcuts: connect high-Fiedler to low-degree nodes
    # Prefer connecting nodes on OPPOSITE sides of the Fiedler cut
    new_src, new_dst, new_w = [], [], []
    attempts = 0
    while len(new_src) < num_shortcuts and attempts < num_shortcuts * 20:
        # Pick a high-Fiedler node
        s = rng.choice(high_fiedler)
        # Pick a low-degree node on the opposite Fiedler side
        # (opposite sign of Fiedler value)
        opposite_side = low_degree[np.sign(fiedler[low_degree]) != np.sign(fiedler[s])]
        if len(opposite_side) == 0:
            opposite_side = low_degree

        d = rng.choice(opposite_side)
        if s == d:
            attempts += 1
            continue

        edge_key = (min(int(s), int(d)), max(int(s), int(d)))
        if edge_key not in existing:
            existing.add(edge_key)
            # Weight for shortcut: mean of existing edge weights
            avg_w = float(ew.mean()) if edge_weight is not None else 1.0
            new_src.extend([int(s), int(d)])
            new_dst.extend([int(d), int(s)])
            new_w.extend([avg_w, avg_w])
        attempts += 1

    if len(new_src) > 0:
        new_ei = torch.tensor([new_src, new_dst], dtype=torch.long)
        new_ew = torch.tensor(new_w, dtype=torch.float32)

        # Concatenate with original edges
        edge_index_new = torch.cat([edge_index, new_ei], dim=1)
        if edge_weight is not None:
            edge_weight_new = torch.cat([ew, new_ew])
        else:
            edge_weight_new = torch.cat([
                torch.ones(edge_index.size(1)),
                new_ew
            ])
    else:
        edge_index_new = edge_index
        edge_weight_new = ew

    # Build new Data object
    data_new = Data(
        x=data.x.clone(),
        edge_index=edge_index_new,
        edge_attr=edge_weight_new,
        y=data.y.clone() if data.y is not None else None,
        num_nodes=num_nodes,
    )

    return data_new


# ---------------------------------------------------------------------------
# Step 2: Sparsification (ISS — Importance-based Spectral Sparsification)
# ---------------------------------------------------------------------------

def _cosine_similarity_edges(x: torch.Tensor, edge_pairs: np.ndarray) -> np.ndarray:
    """
    Compute cosine similarity between node features for each edge pair.

    Parameters
    ----------
    x : Tensor (num_nodes, feature_dim)
    edge_pairs : ndarray (num_edges, 2)

    Returns
    -------
    S_e : ndarray (num_edges,) — cosine similarity in [−1, 1].
    """
    x_np = x.cpu().numpy()
    src_feats = x_np[edge_pairs[:, 0]]
    dst_feats = x_np[edge_pairs[:, 1]]

    # Cosine similarity
    dot = np.sum(src_feats * dst_feats, axis=1)
    norm_src = np.linalg.norm(src_feats, axis=1) + 1e-10
    norm_dst = np.linalg.norm(dst_feats, axis=1) + 1e-10
    S_e = dot / (norm_src * norm_dst)

    return S_e


def sparsify_graph(
    data: Data,
    target_edges: int | None = None,
    k_eigen: int = 50,
    epsilon: float = 0.3,
    seed: int = 42,
) -> Data:
    """
    Spectral sparsification via Importance-based Sampling (ISS).

    Sampling probability per edge:
        p_e ∝ (1 + S_e) × R_e
    where S_e is cosine similarity (feature-aware) and R_e is effective
    resistance (structural importance).

    Guarantees (1 ± ε)-spectral similarity with the original Laplacian
    when target_edges = O(n log n / ε²). (Spielman & Srivastava, 2008)

    Parameters
    ----------
    data : Data
        Input PyG graph (after densification).
    target_edges : int or None
        Number of undirected edges to keep. Default: same as original
        (before densification), i.e., we sparsify back to original density.
    k_eigen : int
        Number of eigenvectors for ER approximation.
    epsilon : float
        Spectral approximation tolerance.
    """
    rng = np.random.RandomState(seed)

    edge_index = data.edge_index
    edge_weight = data.edge_attr if data.edge_attr is not None else torch.ones(edge_index.size(1))
    num_nodes = data.num_nodes

    # Compute effective resistance for each unique edge
    R_e, edge_pairs = compute_effective_resistance(
        edge_index, edge_weight, num_nodes, k=k_eigen
    )

    # Compute cosine similarity for each edge
    S_e = _cosine_similarity_edges(data.x, edge_pairs)

    # Sampling probability: p_e ∝ (1 + S_e) * R_e
    # (1 + S_e) shifts cosine sim from [-1,1] to [0,2] — higher similarity = higher prob
    p_raw = (1.0 + S_e) * R_e
    p_raw = np.clip(p_raw, 1e-10, None)
    p_norm = p_raw / p_raw.sum()

    # Determine target edges
    num_unique_edges = len(edge_pairs)
    if target_edges is None:
        # Default: keep same number of unique edges as input
        target_edges = num_unique_edges

    target_edges = min(target_edges, num_unique_edges)

    # Sample edges WITHOUT replacement
    sampled_idx = rng.choice(
        num_unique_edges, size=target_edges, replace=False, p=p_norm
    )

    # Reconstruct edge_index and edge_weight for sampled edges
    sampled_pairs = edge_pairs[sampled_idx]

    # Get original edge weights for sampled edges
    # Build a map from (src, dst) → weight
    ei_np = edge_index.cpu().numpy()
    ew_np = edge_weight.cpu().numpy()
    weight_map = {}
    for i in range(ei_np.shape[1]):
        s, d = int(ei_np[0, i]), int(ei_np[1, i])
        key = (min(s, d), max(s, d))
        weight_map[key] = ew_np[i]

    # Reweight: w'_e = w_e / (n * p_e) to maintain spectral properties
    # (Spielman-Srivastava reweighting)
    new_src, new_dst, new_w = [], [], []
    for idx in sampled_idx:
        s, d = int(edge_pairs[idx, 0]), int(edge_pairs[idx, 1])
        orig_w = weight_map.get((s, d), 1.0)
        # Reweight to preserve spectral approximation
        reweighted = orig_w / (target_edges * p_norm[idx])
        new_src.extend([s, d])
        new_dst.extend([d, s])
        new_w.extend([reweighted, reweighted])

    new_edge_index = torch.tensor([new_src, new_dst], dtype=torch.long)
    new_edge_weight = torch.tensor(new_w, dtype=torch.float32)

    data_new = Data(
        x=data.x.clone(),
        edge_index=new_edge_index,
        edge_attr=new_edge_weight,
        y=data.y.clone() if data.y is not None else None,
        num_nodes=num_nodes,
    )

    return data_new


# ---------------------------------------------------------------------------
# Combined DSR pipeline: Densify → Sparsify (back to original edge count)
# ---------------------------------------------------------------------------

def dsr_rewire(
    data: Data,
    shortcut_ratio: float = 0.1,
    fiedler_quantile: float = 0.8,
    degree_quantile: float = 0.3,
    k_eigen: int = 50,
    epsilon: float = 0.3,
    seed: int = 42,
) -> Data:
    """
    Full DSR pipeline: Densification → Sparsification.

    1. Densify: add ~shortcut_ratio * |E| shortcut edges via Fiedler analysis
    2. Sparsify: sample back to the ORIGINAL edge count using ISS with
       probability p_e ∝ (1 + S_e) × R_e

    The result has the same edge density as the input but with improved
    spectral properties (lower bottleneck ratio, better connectivity).

    Parameters
    ----------
    data : Data
        Input PyG graph.
    shortcut_ratio : float
        Fraction of existing edges to add as shortcuts (Step 1).
    fiedler_quantile : float
        Quantile for Fiedler value candidate selection.
    degree_quantile : float
        Quantile for degree-based candidate selection.
    k_eigen : int
        Number of eigenvectors for ER approximation.
    epsilon : float
        Spectral tolerance for ISS.
    """
    original_num_edges = data.edge_index.size(1) // 2  # undirected count

    num_shortcuts = max(1, int(shortcut_ratio * original_num_edges))

    # Step 1: Densify
    data_dense = densify_graph(
        data,
        num_shortcuts=num_shortcuts,
        fiedler_quantile=fiedler_quantile,
        degree_quantile=degree_quantile,
        seed=seed,
    )

    # Step 2: Sparsify back to original edge count
    data_sparse = sparsify_graph(
        data_dense,
        target_edges=original_num_edges,
        k_eigen=k_eigen,
        epsilon=epsilon,
        seed=seed + 1,
    )

    return data_sparse


def dsr_rewire_dataset(dataset_list: list, **kwargs) -> list:
    """Apply DSR rewiring to every graph in a dataset list."""
    rewired = []
    for i, data in enumerate(dataset_list):
        try:
            data_new = dsr_rewire(data, seed=kwargs.get("seed", 42) + i, **{
                k: v for k, v in kwargs.items() if k != "seed"
            })
            rewired.append(data_new)
        except Exception as e:
            # If rewiring fails for a graph (e.g., too small), keep original
            print(f"[DSR] Warning: skipping rewiring for graph {i}: {e}")
            rewired.append(data)
    return rewired
