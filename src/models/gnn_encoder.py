"""
gnn_encoder.py
==============
Graph Neural Network encoder for 1-D scattering difference signals.

Approach
--------
Each 1-D spectrum is treated as a **sequence of L nodes** arranged along
a physical q-axis.  A k-nearest-neighbour graph is constructed in the
feature space ``(intensity, normalised_position)`` so that nodes with
similar local scattering behaviour are connected, not just spatially
adjacent ones.

The graph is then processed by stacked **GATv2Conv** layers (attention-
based message passing) that capture both local and non-local correlations
in the difference signal before global pooling produces a compact embedding
``z ∈ ℝ^D``.

Graph construction
------------------
* Node features: ``[intensity, q_position]`` – shape ``(L, 2)``.
* Edges: k-NN in the 2-D feature space; made undirected for symmetric
  message passing.

GATv2Conv layers
-----------------
All layers except the last use ``heads > 1`` (multi-head attention).
The last layer uses a single head so that ``per_head_dim == latent_dim``.

Pooling
-------
Both ``global_mean_pool`` and ``global_max_pool`` are concatenated to give
a ``2D``-dimensional vector, which is projected back to ``D`` via a linear
layer:  ``z = W [mean‖max]``.

References
----------
- GATv2: Brody et al., "How Attentive are Graph Attention Networks?",
  ICLR 2022.  https://arxiv.org/abs/2105.14491
- PyG API: https://pytorch-geometric.readthedocs.io
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Batch, Data
from torch_geometric.nn import GATv2Conv, global_max_pool, global_mean_pool
from torch_geometric.utils import to_undirected


# ---------------------------------------------------------------------------
# Graph construction utilities
# ---------------------------------------------------------------------------

def _build_knn_edges(node_features: torch.Tensor, k: int) -> torch.Tensor:
    """Construct a k-NN edge index for a single graph in feature space.

    Distances are computed as the Euclidean distance between node feature
    vectors ``(intensity, position)``.  Self-loops are excluded by filling
    the diagonal of the distance matrix with ``+inf`` before selecting the
    k smallest values.

    The resulting directed edges are made undirected via
    :func:`torch_geometric.utils.to_undirected`, so each pair of connected
    nodes has edges in both directions (required by GATv2Conv's asymmetric
    attention).

    Args:
        node_features: Node feature matrix of shape ``(L, F)`` where *L* is
            the number of nodes (spectrum length) and *F* is the feature
            dimension (2 by default: intensity + q-position).
        k: Number of nearest neighbours per node.  Clamped to ``L − 1``
            to avoid requesting more neighbours than available.

    Returns:
        Edge index tensor of shape ``(2, E)`` in COO format, where *E* is
        the number of (undirected) edges.
    """
    num_nodes = node_features.shape[0]
    # Clamp k so we never request more neighbours than possible
    k = min(k, num_nodes - 1)

    # Compute all pairwise Euclidean distances: (L, L)
    # Using torch.no_grad() because edge construction is not part of the
    # differentiable computation graph (graph topology is fixed per forward pass)
    with torch.no_grad():
        dist = torch.cdist(node_features, node_features, p=2)
        # Remove self-distances so a node is not its own neighbour
        dist.fill_diagonal_(float("inf"))
        # Select the k smallest distances per row: (L, k) indices
        knn_indices = dist.topk(k, largest=False).indices

    # Build source and target index tensors
    src = (
        torch.arange(num_nodes, device=node_features.device)
        .unsqueeze(1)               # (L, 1)
        .expand(-1, k)              # (L, k)
        .reshape(-1)                # (L*k,)
    )
    dst = knn_indices.reshape(-1)   # (L*k,)

    # Make undirected: each directed edge (u→v) gains a reverse edge (v→u)
    return to_undirected(torch.stack([src, dst]), num_nodes=num_nodes)


def _spectra_to_pyg_batch(x_diff: torch.Tensor, k: int) -> Batch:
    """Convert a batch of 1-D difference signals into a PyG :class:`Batch`.

    Each spectrum becomes an individual graph with:

    * ``x`` – node feature matrix ``(L, 2)``: [intensity, q_position]
    * ``edge_index`` – k-NN edge index ``(2, E)``

    Positions are normalised to ``[0, 1]`` so both feature dimensions are
    on a comparable scale.

    Args:
        x_diff: Batch of difference signals of shape ``(B, L)``.
        k: Number of nearest neighbours for graph construction.

    Returns:
        A :class:`~torch_geometric.data.Batch` containing *B* graphs.
    """
    batch_size, seq_len = x_diff.shape

    # Normalised position vector shared across all samples in the batch
    q_pos = torch.linspace(0.0, 1.0, seq_len, device=x_diff.device).unsqueeze(1)  # (L, 1)

    data_list: list[Data] = []
    for i in range(batch_size):
        # Node features: stack [intensity_i, q_position] → (L, 2)
        node_feats = torch.cat([x_diff[i].unsqueeze(1), q_pos], dim=1)
        edge_index = _build_knn_edges(node_feats, k)
        data_list.append(Data(x=node_feats, edge_index=edge_index))

    return Batch.from_data_list(data_list)


# ---------------------------------------------------------------------------
# GNN Encoder
# ---------------------------------------------------------------------------

class GNNEncoder(nn.Module):
    """Graph Attention Network encoder for scattering difference signals.

    Converts a batch of 1-D difference spectra ``x_diff ∈ ℝ^(B×L)`` into
    a batch of latent embeddings ``z ∈ ℝ^(B×D)`` via:

    1. Build a k-NN graph per sample (feature-space proximity).
    2. Apply *num_layers* GATv2Conv layers with ELU activation + LayerNorm.
    3. Pool: concatenate global mean-pool and global max-pool → ``(B, 2·H)``.
    4. Project: linear ``(2·H → D)`` → output embedding ``z``.

    Args:
        signal_len: Length *L* of the input spectrum (300 for SAXS, 290 for
            WAXS).  Stored for reference; not used in the computation.
        node_in_dim: Dimensionality of the initial node feature vectors
            (default 2: intensity + normalised q-position).
        hidden_dim: Width of the intermediate GATv2Conv layers (after
            concatenating all attention heads).
        latent_dim: Dimensionality *D* of the output embedding.
        num_layers: Number of GATv2Conv layers.
        k_neighbors: *k* in the k-NN graph construction.
        dropout: Dropout probability applied after each layer.
        heads: Number of attention heads for all layers except the last
            (which always uses 1 head to produce exactly *latent_dim* dims).

    Example::

        enc = GNNEncoder(signal_len=300, latent_dim=64)
        z = enc(x_diff_s)   # x_diff_s: (B, 300)  →  z: (B, 64)
    """

    def __init__(
        self,
        signal_len: int,
        node_in_dim: int = 2,
        hidden_dim: int = 128,
        latent_dim: int = 64,
        num_layers: int = 3,
        k_neighbors: int = 8,
        dropout: float = 0.1,
        heads: int = 4,
    ) -> None:
        super().__init__()
        self.k_neighbors = k_neighbors
        self.dropout = dropout

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()

        in_dim = node_in_dim
        for layer_idx in range(num_layers):
            is_last_layer = layer_idx == num_layers - 1

            # Last layer: single head so output width == latent_dim exactly
            n_heads = 1 if is_last_layer else heads
            out_width = latent_dim if is_last_layer else hidden_dim
            # Each head outputs (out_width // n_heads) dims; concat gives out_width
            per_head_dim = max(1, out_width // n_heads)

            self.convs.append(
                GATv2Conv(
                    in_channels=in_dim,
                    out_channels=per_head_dim,
                    heads=n_heads,
                    dropout=dropout,
                    concat=True,            # concatenate head outputs
                    add_self_loops=True,    # include self-information
                )
            )
            # LayerNorm over the concatenated head outputs
            self.norms.append(nn.LayerNorm(per_head_dim * n_heads))
            in_dim = per_head_dim * n_heads  # input dim for next layer

        # Pool projection: mean‖max concatenation doubles the width
        self.pool_proj = nn.Linear(in_dim * 2, latent_dim)

    def forward(self, x_diff: torch.Tensor) -> torch.Tensor:
        """Encode a batch of difference signals into latent embeddings.

        Args:
            x_diff: Difference signal batch of shape ``(B, L)``.

        Returns:
            Latent embedding ``z`` of shape ``(B, latent_dim)``.
        """
        # Convert dense spectra to a PyG mini-batch of graphs
        pyg_batch = _spectra_to_pyg_batch(x_diff, self.k_neighbors)
        h = pyg_batch.x               # (B*L, node_in_dim) – all nodes stacked
        edge_index = pyg_batch.edge_index
        batch_vector = pyg_batch.batch  # node-to-graph assignment: (B*L,)

        # Message-passing layers
        for conv, norm in zip(self.convs, self.norms):
            # GATv2 message passing → ELU activation → LayerNorm
            h = norm(F.elu(conv(h, edge_index)))
            # Stochastic depth regularisation during training
            h = F.dropout(h, p=self.dropout, training=self.training)

        # Readout: combine mean and max pooling for richer global representation
        z_mean = global_mean_pool(h, batch_vector)   # (B, in_dim)
        z_max = global_max_pool(h, batch_vector)      # (B, in_dim)
        z = self.pool_proj(torch.cat([z_mean, z_max], dim=-1))  # (B, latent_dim)

        return z
