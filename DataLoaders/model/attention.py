"""
model/attention.py
──────────────────
T5-style multi-head attention with:
  - Relative position bias (encoder self-attention + decoder self-attention)
  - Standard cross-attention (no positional bias)
  - Efficient scaled dot-product attention (uses torch F.scaled_dot_product_attention
    when available for Flash Attention support)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class T5RelativePositionBias(nn.Module):
    """
    Learned relative position bias (Raffel et al. 2020, Appendix B).
    Maps relative distance → bucket → learned scalar bias added to attention logits.
    """

    def __init__(self, num_heads: int, num_buckets: int = 32, max_distance: int = 128):
        super().__init__()
        self.num_heads    = num_heads
        self.num_buckets  = num_buckets
        self.max_distance = max_distance
        self.embedding    = nn.Embedding(num_buckets, num_heads)

    @staticmethod
    def _relative_position_bucket(
        relative_position: torch.Tensor,
        bidirectional: bool = True,
        num_buckets: int = 32,
        max_distance: int = 128,
    ) -> torch.Tensor:
        """Converts integer relative positions to bucket indices."""
        ret = 0
        n   = -relative_position

        if bidirectional:
            num_buckets //= 2
            ret         += (n < 0).to(torch.long) * num_buckets
            n            = n.abs()
        else:
            n = n.clamp(min=0)

        max_exact = num_buckets // 2
        is_small  = n < max_exact

        val_if_large = max_exact + (
            torch.log(n.float() / max_exact)
            / math.log(max_distance / max_exact)
            * (num_buckets - max_exact)
        ).to(torch.long).clamp(max=num_buckets - 1)

        return ret + torch.where(is_small, n, val_if_large)

    def forward(self, seq_len_q: int, seq_len_k: int,
                bidirectional: bool = True, device: torch.device = None) -> torch.Tensor:
        """
        Returns bias of shape (1, num_heads, seq_len_q, seq_len_k)
        to be added to attention logits.
        """
        q_pos = torch.arange(seq_len_q, dtype=torch.long, device=device).unsqueeze(1)
        k_pos = torch.arange(seq_len_k, dtype=torch.long, device=device).unsqueeze(0)
        rel   = k_pos - q_pos   # (seq_q, seq_k)

        buckets = self._relative_position_bucket(
            rel, bidirectional=bidirectional,
            num_buckets=self.num_buckets,
            max_distance=self.max_distance,
        )                                            # (seq_q, seq_k)
        bias = self.embedding(buckets)               # (seq_q, seq_k, heads)
        bias = bias.permute(2, 0, 1).unsqueeze(0)   # (1, heads, seq_q, seq_k)
        return bias


# ─────────────────────────────────────────────────────────────────────────────

class MultiHeadAttention(nn.Module):
    """
    Unified multi-head attention for:
      mode='self'  — encoder or decoder self-attention (with relative bias)
      mode='cross' — decoder cross-attention          (no relative bias)
    """

    def __init__(
        self,
        d_model:      int,
        num_heads:    int,
        d_kv:         int,
        dropout:      float = 0.0,
        mode:         str   = "self",   # "self" or "cross"
        num_buckets:  int   = 32,
        max_distance: int   = 128,
    ):
        super().__init__()
        assert mode in ("self", "cross")
        self.mode      = mode
        self.num_heads = num_heads
        self.d_kv      = d_kv
        self.scale     = d_kv ** -0.5

        inner = num_heads * d_kv
        self.q = nn.Linear(d_model, inner, bias=False)
        self.k = nn.Linear(d_model, inner, bias=False)
        self.v = nn.Linear(d_model, inner, bias=False)
        self.o = nn.Linear(inner, d_model, bias=False)

        self.attn_dropout = nn.Dropout(dropout)

        if mode == "self":
            self.rel_bias = T5RelativePositionBias(num_heads, num_buckets, max_distance)
        else:
            self.rel_bias = None

    def forward(
        self,
        query:              torch.Tensor,                  # (B, Lq, D)
        key_value:          torch.Tensor,                  # (B, Lk, D)
        key_padding_mask:   Optional[torch.Tensor] = None, # (B, Lk)  1=keep 0=mask
        causal_mask:        bool = False,
    ) -> torch.Tensor:
        B, Lq, _ = query.shape
        Lk       = key_value.shape[1]

        # ── project ──────────────────────────────────────────────────────────
        Q = self.q(query).view(B, Lq, self.num_heads, self.d_kv).transpose(1, 2)   # (B, H, Lq, d_kv)
        K = self.k(key_value).view(B, Lk, self.num_heads, self.d_kv).transpose(1, 2)
        V = self.v(key_value).view(B, Lk, self.num_heads, self.d_kv).transpose(1, 2)

        # ── attention scores ─────────────────────────────────────────────────
        scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale   # (B, H, Lq, Lk)

        # relative position bias (self-attention only)
        if self.rel_bias is not None:
            bidirectional = not causal_mask
            scores = scores + self.rel_bias(Lq, Lk, bidirectional=bidirectional, device=query.device)

        # causal mask
        if causal_mask:
            causal = torch.ones(Lq, Lk, dtype=torch.bool, device=query.device).triu(diagonal=1)
            scores = scores.masked_fill(causal.unsqueeze(0).unsqueeze(0), float("-inf"))

        # key padding mask
        if key_padding_mask is not None:
            pad_mask = (key_padding_mask == 0)          # True = ignore
            scores   = scores.masked_fill(pad_mask.unsqueeze(1).unsqueeze(2), float("-inf"))

        # ── softmax + dropout ────────────────────────────────────────────────
        attn_weights = F.softmax(scores.float(), dim=-1).to(Q.dtype)
        attn_weights = self.attn_dropout(attn_weights)

        # ── output ───────────────────────────────────────────────────────────
        out = torch.matmul(attn_weights, V)                             # (B, H, Lq, d_kv)
        out = out.transpose(1, 2).contiguous().view(B, Lq, self.num_heads * self.d_kv)
        return self.o(out)
