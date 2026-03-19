"""
GNN Encoder — PyTorch Geometric GATv2Conv.
Each 1-D spectrum → sequence of L nodes → k-NN graph → GATv2 layers → z ∈ R^D.
"""
from __future__ import annotations
import torch, torch.nn as nn, torch.nn.functional as F
from torch_geometric.data import Data, Batch
from torch_geometric.nn  import GATv2Conv, global_mean_pool, global_max_pool
from torch_geometric.utils import to_undirected

def _build_edges(node_feats: torch.Tensor, k: int) -> torch.Tensor:
    L  = node_feats.shape[0]
    k  = min(k, L - 1)
    with torch.no_grad():
        dist = torch.cdist(node_feats, node_feats, p=2)
        dist.fill_diagonal_(float("inf"))
        knn  = dist.topk(k, largest=False).indices        # (L, k)
    row = torch.arange(L, device=node_feats.device).unsqueeze(1).expand(-1, k).reshape(-1)
    col = knn.reshape(-1)
    return to_undirected(torch.stack([row, col]), num_nodes=L)

def _to_pyg_batch(x_diff: torch.Tensor, k: int) -> Batch:
    B, L   = x_diff.shape
    pos    = torch.linspace(0, 1, L, device=x_diff.device).unsqueeze(1)
    data_list = []
    for i in range(B):
        nf  = torch.cat([x_diff[i].unsqueeze(1), pos], dim=1)   # (L, 2)
        ei  = _build_edges(nf, k)
        data_list.append(Data(x=nf, edge_index=ei))
    return Batch.from_data_list(data_list)

class GNNEncoder(nn.Module):
    """
    Args:
        signal_len  : L (300 for SAXS, 290 for WAXS)
        node_in_dim : 2  (intensity + normalised position)
        hidden_dim  : width of GATv2 layers
        latent_dim  : D — output embedding size
        num_layers  : depth
        k_neighbors : k for k-NN graph
        dropout     : dropout probability
        heads       : attention heads
    """
    def __init__(self, signal_len, node_in_dim=2, hidden_dim=128,
                 latent_dim=64, num_layers=3, k_neighbors=8,
                 dropout=0.1, heads=4):
        super().__init__()
        self.k  = k_neighbors
        self.dp = dropout
        self.convs, self.norms = nn.ModuleList(), nn.ModuleList()
        in_d = node_in_dim
        for i in range(num_layers):
            is_last   = (i == num_layers - 1)
            n_heads   = 1 if is_last else heads
            per_head  = max(1, (latent_dim if is_last else hidden_dim) // n_heads)
            self.convs.append(GATv2Conv(in_d, per_head, heads=n_heads,
                                        dropout=dropout, concat=True, add_self_loops=True))
            self.norms.append(nn.LayerNorm(per_head * n_heads))
            in_d = per_head * n_heads
        self.pool_proj = nn.Linear(in_d * 2, latent_dim)

    def forward(self, x_diff: torch.Tensor) -> torch.Tensor:
        pg  = _to_pyg_batch(x_diff, self.k)
        h, ei, bv = pg.x, pg.edge_index, pg.batch
        for conv, norm in zip(self.convs, self.norms):
            h = norm(F.elu(conv(h, ei)))
            h = F.dropout(h, p=self.dp, training=self.training)
        z = self.pool_proj(torch.cat([global_mean_pool(h, bv),
                                      global_max_pool(h, bv)], dim=-1))
        return z
