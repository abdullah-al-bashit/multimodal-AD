"""
cross_modal.py
==============
Cross-modal fusion module: ``Z_mp = fuse(z_S, z_W)``.

Purpose
-------
After the two GNN encoders independently produce per-modality embeddings
``z_S ∈ ℝ^(B×D)`` (SAXS) and ``z_W ∈ ℝ^(B×D)`` (WAXS), the
:class:`CrossModalFusion` module merges them into a single multimodal
representation ``Z_mp ∈ ℝ^(B×D)``.

This shared representation is supervised by the InfoNCE contrastive loss
``L_NCE``, which encourages ``z_S`` and ``z_W`` from the *same* sample to
be more similar than embeddings from different samples.

Fusion strategies
-----------------
Three fusion methods are supported:

``"attention"`` (default)
    Bidirectional cross-modal attention using :class:`~torch.nn.MultiheadAttention`.
    ``z_S`` attends to ``z_W`` (and vice versa); the resulting context vectors
    are concatenated and projected: ``Z_mp = W [c_{S←W} ‖ c_{W←S}]``.
    This is the most expressive option and handles asymmetric modality
    relationships.

``"concat"``
    Simple concatenation followed by an MLP projection:
    ``Z_mp = W [z_S ‖ z_W]``.  Efficient and often competitive.

``"add"``
    Element-wise addition: ``Z_mp = z_S + z_W``.  Fastest option;
    assumes equal contribution from both modalities.
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossModalFusion(nn.Module):
    """Bidirectional cross-modal fusion of SAXS and WAXS embeddings.

    Args:
        latent_dim: Dimensionality *D* of each input embedding ``z_S`` and
            ``z_W``.
        method: Fusion strategy – ``"attention"``, ``"concat"``, or
            ``"add"``.  See module docstring for details.
        out_dim: Dimensionality of the fused output ``Z_mp``.  Defaults to
            ``latent_dim`` if not specified.

    Raises:
        ValueError: If *method* is not one of the three supported options.

    Example::

        fusion = CrossModalFusion(latent_dim=64, method="attention")
        z_mp = fusion(z_s, z_w)   # z_mp: (B, 64)
    """

    _SUPPORTED_METHODS = frozenset({"attention", "concat", "add"})

    def __init__(
        self,
        latent_dim: int = 64,
        method: str = "attention",
        out_dim: Optional[int] = None,
    ) -> None:
        super().__init__()

        if method not in self._SUPPORTED_METHODS:
            raise ValueError(
                f"Unknown fusion method '{method}'. "
                f"Choose from {self._SUPPORTED_METHODS}."
            )

        self.method = method
        out_dim = out_dim or latent_dim

        if method == "attention":
            # Bidirectional cross-attention: S attends to W, and W attends to S
            self.attn_s2w = nn.MultiheadAttention(
                embed_dim=latent_dim, num_heads=4, batch_first=True
            )
            self.attn_w2s = nn.MultiheadAttention(
                embed_dim=latent_dim, num_heads=4, batch_first=True
            )
            # Project concatenated context vectors [c_{S←W} ‖ c_{W←S}] → Z_mp
            self.proj = nn.Sequential(
                nn.Linear(latent_dim * 2, out_dim),
                nn.LayerNorm(out_dim),
                nn.GELU(),
            )

        elif method == "concat":
            # Simple MLP projection of the concatenated pair
            self.proj = nn.Sequential(
                nn.Linear(latent_dim * 2, out_dim),
                nn.LayerNorm(out_dim),
                nn.GELU(),
            )

        else:  # "add"
            # Element-wise sum requires equal dimensions; no learnable params
            self.proj = nn.Identity()

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, z_s: torch.Tensor, z_w: torch.Tensor) -> torch.Tensor:
        """Fuse SAXS and WAXS embeddings into a single multimodal representation.

        Args:
            z_s: SAXS embedding of shape ``(B, D)``.
            z_w: WAXS embedding of shape ``(B, D)``.

        Returns:
            Fused multimodal embedding ``Z_mp`` of shape ``(B, out_dim)``.
        """
        if self.method == "attention":
            return self._attention_fusion(z_s, z_w)
        if self.method == "concat":
            return self.proj(torch.cat([z_s, z_w], dim=-1))
        # "add": element-wise sum (proj is nn.Identity)
        return self.proj(z_s + z_w)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _attention_fusion(
        self, z_s: torch.Tensor, z_w: torch.Tensor
    ) -> torch.Tensor:
        """Bidirectional cross-attention fusion.

        Each embedding acts as the query for the other modality's key/value,
        allowing each modality to selectively attend to the other.

        Args:
            z_s: SAXS embedding ``(B, D)`` – will be treated as a 1-token
                sequence ``(B, 1, D)`` for the attention API.
            z_w: WAXS embedding ``(B, D)``.

        Returns:
            Projected fused embedding of shape ``(B, out_dim)``.
        """
        # Reshape to sequence format required by MultiheadAttention: (B, 1, D)
        q_s = z_s.unsqueeze(1)
        q_w = z_w.unsqueeze(1)

        # c_{S←W}: SAXS queries attend to WAXS keys/values
        c_s, _ = self.attn_s2w(query=q_s, key=q_w, value=q_w)
        # c_{W←S}: WAXS queries attend to SAXS keys/values
        c_w, _ = self.attn_w2s(query=q_w, key=q_s, value=q_s)

        # Squeeze the sequence dimension back: (B, 1, D) → (B, D)
        c_s = c_s.squeeze(1)
        c_w = c_w.squeeze(1)

        # Concatenate and project: (B, 2D) → (B, out_dim)
        return self.proj(torch.cat([c_s, c_w], dim=-1))
