"""Attention module for LTX-2."""

import math
from typing import Optional, Tuple

import mlx.core as mx
import mlx.nn as nn

from mlx_video.models.ltx_2.config import LTXRopeType
from mlx_video.models.ltx_2.rope import apply_rotary_emb


def scaled_dot_product_attention(
    q: mx.array,
    k: mx.array,
    v: mx.array,
    heads: int,
    mask: Optional[mx.array] = None,
) -> mx.array:

    b, q_seq_len, dim = q.shape
    _, kv_seq_len, _ = k.shape
    dim_head = dim // heads

    # Reshape to (B, seq_len, heads, dim_head)
    q = mx.reshape(q, (b, q_seq_len, heads, dim_head))
    k = mx.reshape(k, (b, kv_seq_len, heads, dim_head))
    v = mx.reshape(v, (b, kv_seq_len, heads, dim_head))

    # Transpose to (B, heads, seq_len, dim_head)
    q = mx.swapaxes(q, 1, 2)
    k = mx.swapaxes(k, 1, 2)
    v = mx.swapaxes(v, 1, 2)

    # Handle mask dimensions
    if mask is not None:
        # Add batch dimension if needed
        if mask.ndim == 2:
            mask = mx.expand_dims(mask, axis=0)
        # Add heads dimension if needed
        if mask.ndim == 3:
            mask = mx.expand_dims(mask, axis=1)

    # Compute scaled dot-product attention
    scale = 1.0 / math.sqrt(dim_head)

    out = mx.fast.scaled_dot_product_attention(q, k, v, scale=scale, mask=mask)

    # Reshape back to (B, q_seq_len, heads * dim_head)
    out = mx.swapaxes(out, 1, 2)
    out = mx.reshape(out, (b, q_seq_len, heads * dim_head))

    return out


class Attention(nn.Module):
    """Multi-head attention with rotary position embeddings.

    Supports both self-attention and cross-attention.
    """

    def __init__(
        self,
        query_dim: int,
        context_dim: Optional[int] = None,
        heads: int = 8,
        dim_head: int = 64,
        norm_eps: float = 1e-6,
        rope_type: LTXRopeType = LTXRopeType.INTERLEAVED,
        has_gate_logits: bool = False,
    ):
        super().__init__()

        self.rope_type = rope_type
        self.heads = heads
        self.dim_head = dim_head

        inner_dim = dim_head * heads
        context_dim = query_dim if context_dim is None else context_dim

        # Q, K, V projections
        self.to_q = nn.Linear(query_dim, inner_dim, bias=True)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=True)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=True)

        # Q and K normalization
        self.q_norm = nn.RMSNorm(inner_dim, eps=norm_eps)
        self.k_norm = nn.RMSNorm(inner_dim, eps=norm_eps)

        # Output projection
        self.to_out = nn.Linear(inner_dim, query_dim, bias=True)

        # Per-head gating (LTX-2.3)
        if has_gate_logits:
            self.to_gate_logits = nn.Linear(query_dim, heads, bias=True)

    def __call__(
        self,
        x: mx.array,
        context: Optional[mx.array] = None,
        mask: Optional[mx.array] = None,
        pe: Optional[Tuple[mx.array, mx.array]] = None,
        k_pe: Optional[Tuple[mx.array, mx.array]] = None,
        skip_attention: bool = False,
    ) -> mx.array:
        """Forward pass.

        Args:
            x: Query input of shape (B, seq_len, query_dim)
            context: Context for cross-attention. If None, uses x (self-attention)
            mask: Attention mask
            pe: Position embeddings for query (and key if k_pe is None)
            k_pe: Position embeddings for key (optional, uses pe if None)
            skip_attention: If True, bypass Q*K*V attention and use value projection
                only (for STG perturbation). Matches PyTorch all_perturbed=True.

        Returns:
            Attention output of shape (B, seq_len, query_dim)
        """
        # Compute per-head gate early (from original input)
        gate = None
        if hasattr(self, "to_gate_logits"):
            gate = 2.0 * mx.sigmoid(self.to_gate_logits(x))  # (B, seq, heads)

        context = x if context is None else context
        v = self.to_v(context)

        if skip_attention:
            # STG: bypass Q*K*V attention, use value projection only
            out = v
        else:
            # Standard attention
            q = self.to_q(x)
            k = self.to_k(context)

            q = self.q_norm(q)
            k = self.k_norm(k)

            if pe is not None:
                q = apply_rotary_emb(q, pe, self.rope_type)
                k_pe_to_use = pe if k_pe is None else k_pe
                k = apply_rotary_emb(k, k_pe_to_use, self.rope_type)

            out = scaled_dot_product_attention(q, k, v, self.heads, mask)

        # Apply per-head gating
        if gate is not None:
            b, seq_len, _ = out.shape
            out = mx.reshape(out, (b, seq_len, self.heads, self.dim_head))
            out = out * gate[..., None]
            out = mx.reshape(out, (b, seq_len, -1))

        # Project output
        return self.to_out(out)
