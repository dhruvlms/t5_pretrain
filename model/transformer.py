"""
model/transformer.py
────────────────────
Full T5-style Encoder-Decoder Transformer (~60 M parameters).

Architecture details:
  - Shared token embeddings between encoder and decoder
  - Pre-norm (RMSNorm before each sub-layer)
  - T5 relative position bias in self-attention (not cross-attention)
  - No absolute positional embeddings
  - Bias-free projections throughout
  - Encoder: 6 layers × (self-attn + FFN)
  - Decoder: 6 layers × (masked self-attn + cross-attn + FFN)
  - Output head: embedding weight tying (LM head = transpose of embedding)

Parameter count (approx):
  Embedding table:   32128 × 512               = 16.5 M
  Encoder (6 layers):  each ~3.1 M             = 18.7 M
  Decoder (6 layers):  each ~4.1 M             = 24.8 M
  ─────────────────────────────────────────────────────
  Total                                        ≈ 60.0 M   (weight tying saves ~16 M vs separate LM head)
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

from config import ModelConfig
from model.attention import MultiHeadAttention
from model.feedforward import RMSNorm, TransformerFFN


# ─────────────────────────────────────────────────────────────────────────────
#  Encoder layer
# ─────────────────────────────────────────────────────────────────────────────
class EncoderLayer(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.self_attn = MultiHeadAttention(
            d_model     = cfg.d_model,
            num_heads   = cfg.num_heads,
            d_kv        = cfg.d_kv,
            dropout     = cfg.dropout,
            mode        = "self",
            num_buckets = cfg.relative_attn_buckets,
            max_distance= cfg.relative_attn_max_distance,
        )
        self.attn_norm = RMSNorm(cfg.d_model)
        self.ffn       = TransformerFFN(cfg.d_model, cfg.d_ff, cfg.dropout)
        self.dropout   = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor, src_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # pre-norm self-attention + residual
        normed = self.attn_norm(x)
        x      = x + self.dropout(self.self_attn(normed, normed, key_padding_mask=src_mask))
        # pre-norm FFN + residual
        x      = self.ffn(x)
        return x


# ─────────────────────────────────────────────────────────────────────────────
#  Decoder layer
# ─────────────────────────────────────────────────────────────────────────────
class DecoderLayer(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        # masked self-attention
        self.self_attn      = MultiHeadAttention(
            d_model     = cfg.d_model,
            num_heads   = cfg.num_heads,
            d_kv        = cfg.d_kv,
            dropout     = cfg.dropout,
            mode        = "self",
            num_buckets = cfg.relative_attn_buckets,
            max_distance= cfg.relative_attn_max_distance,
        )
        self.self_attn_norm = RMSNorm(cfg.d_model)

        # cross-attention
        self.cross_attn      = MultiHeadAttention(
            d_model  = cfg.d_model,
            num_heads= cfg.num_heads,
            d_kv     = cfg.d_kv,
            dropout  = cfg.dropout,
            mode     = "cross",
        )
        self.cross_attn_norm = RMSNorm(cfg.d_model)

        self.ffn     = TransformerFFN(cfg.d_model, cfg.d_ff, cfg.dropout)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(
        self,
        x:            torch.Tensor,
        enc_out:      torch.Tensor,
        tgt_mask:     Optional[torch.Tensor] = None,   # (B, Lk) padding mask for decoder
        src_mask:     Optional[torch.Tensor] = None,   # (B, Ls) padding mask for encoder
    ) -> torch.Tensor:
        # masked self-attention (causal)
        normed = self.self_attn_norm(x)
        x = x + self.dropout(self.self_attn(normed, normed, key_padding_mask=tgt_mask, causal_mask=True))

        # cross-attention
        normed = self.cross_attn_norm(x)
        x = x + self.dropout(self.cross_attn(normed, enc_out, key_padding_mask=src_mask))

        # FFN
        x = self.ffn(x)
        return x


# ─────────────────────────────────────────────────────────────────────────────
#  Full Encoder-Decoder
# ─────────────────────────────────────────────────────────────────────────────
class T5Model(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg

        # Shared embedding (encoder + decoder share weights)
        self.shared_embed = nn.Embedding(cfg.vocab_size, cfg.d_model, padding_idx=cfg.pad_id)

        # Encoder
        self.encoder_layers   = nn.ModuleList([EncoderLayer(cfg) for _ in range(cfg.num_encoder_layers)])
        self.encoder_norm     = RMSNorm(cfg.d_model)

        # Decoder
        self.decoder_layers   = nn.ModuleList([DecoderLayer(cfg) for _ in range(cfg.num_decoder_layers)])
        self.decoder_norm     = RMSNorm(cfg.d_model)

        self.dropout = nn.Dropout(cfg.dropout)

        # LM head — weight tying with embedding
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.lm_head.weight = self.shared_embed.weight   # weight tying

        self._init_weights()

    def _init_weights(self):
        """T5-style weight initialization."""
        std = self.cfg.d_model ** -0.5
        nn.init.normal_(self.shared_embed.weight, mean=0.0, std=std)
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=std)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding) and module is not self.shared_embed:
                nn.init.normal_(module.weight, mean=0.0, std=std)

    # ── Encoder ──────────────────────────────────────────────────────────────
    def encode(
        self,
        input_ids:      torch.Tensor,             # (B, Ls)
        attention_mask: Optional[torch.Tensor],   # (B, Ls)  1=keep
    ) -> torch.Tensor:
        x = self.dropout(self.shared_embed(input_ids))   # (B, Ls, D)
        for layer in self.encoder_layers:
            x = layer(x, src_mask=attention_mask)
        return self.encoder_norm(x)                       # (B, Ls, D)

    # ── Decoder ──────────────────────────────────────────────────────────────
    def decode(
        self,
        decoder_input_ids:      torch.Tensor,              # (B, Lt)
        encoder_hidden_states:  torch.Tensor,              # (B, Ls, D)
        decoder_attention_mask: Optional[torch.Tensor],    # (B, Lt)
        encoder_attention_mask: Optional[torch.Tensor],    # (B, Ls)
    ) -> torch.Tensor:
        x = self.dropout(self.shared_embed(decoder_input_ids))  # (B, Lt, D)
        for layer in self.decoder_layers:
            x = layer(
                x,
                enc_out  = encoder_hidden_states,
                tgt_mask = decoder_attention_mask,
                src_mask = encoder_attention_mask,
            )
        return self.decoder_norm(x)                              # (B, Lt, D)

    # ── Forward ──────────────────────────────────────────────────────────────
    def forward(
        self,
        input_ids:              torch.Tensor,
        attention_mask:         Optional[torch.Tensor] = None,
        decoder_input_ids:      Optional[torch.Tensor] = None,
        decoder_attention_mask: Optional[torch.Tensor] = None,
        labels:                 Optional[torch.Tensor] = None,
    ) -> dict:
        """
        Args:
            input_ids              (B, Ls)  — corrupted encoder input
            attention_mask         (B, Ls)  — 1=real token, 0=pad
            decoder_input_ids      (B, Lt)  — target shifted right (teacher forcing)
            decoder_attention_mask (B, Lt)
            labels                 (B, Lt)  — target token ids; -100 = ignore

        Returns dict with:
            loss    : scalar cross-entropy (if labels provided)
            logits  : (B, Lt, vocab_size)
        """
        enc_out = self.encode(input_ids, attention_mask)

        dec_out = self.decode(
            decoder_input_ids      = decoder_input_ids,
            encoder_hidden_states  = enc_out,
            decoder_attention_mask = decoder_attention_mask,
            encoder_attention_mask = attention_mask,
        )

        logits = self.lm_head(dec_out)   # (B, Lt, V)

        loss = None
        if labels is not None:
            # Flatten and compute cross-entropy; labels=-100 are ignored
            loss = F.cross_entropy(
                logits.view(-1, self.cfg.vocab_size),
                labels.view(-1),
                ignore_index = -100,
            )

        return {"loss": loss, "logits": logits}

    # ── Greedy decode (for evaluation / qualitative checks) ──────────────────
    @torch.no_grad()
    def generate(
        self,
        input_ids:      torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        max_new_tokens: int = 64,
    ) -> torch.Tensor:
        """Greedy auto-regressive generation.  Returns (B, T) token ids."""
        enc_out = self.encode(input_ids, attention_mask)
        B       = input_ids.shape[0]
        device  = input_ids.device

        # start with <extra_id_0> as the first decoder token (T5 convention)
        from config import VOCAB_SIZE, NUM_SENTINEL_TOKENS, EOS_ID, PAD_ID
        bos_id  = VOCAB_SIZE - 1  # <extra_id_0>
        eos_id  = EOS_ID

        dec_ids = torch.full((B, 1), bos_id, dtype=torch.long, device=device)
        finished = torch.zeros(B, dtype=torch.bool, device=device)

        for _ in range(max_new_tokens):
            dec_mask = torch.ones_like(dec_ids)
            dec_out  = self.decode(dec_ids, enc_out, dec_mask, attention_mask)
            next_tok = self.lm_head(dec_out[:, -1, :]).argmax(dim=-1)  # (B,)
            dec_ids  = torch.cat([dec_ids, next_tok.unsqueeze(1)], dim=1)
            finished |= (next_tok == eos_id)
            if finished.all():
                break

        return dec_ids[:, 1:]   # strip leading BOS


# ─────────────────────────────────────────────────────────────────────────────
#  Parameter count utility
# ─────────────────────────────────────────────────────────────────────────────
def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    from config import MODEL_CFG
    m = T5Model(MODEL_CFG)
    n = count_parameters(m)
    print(f"Model parameters: {n:,}  ({n/1e6:.1f} M)")
    # quick forward pass
    B, Ls, Lt = 2, 32, 16
    ids  = torch.randint(0, 100, (B, Ls))
    dids = torch.randint(0, 100, (B, Lt))
    labs = torch.randint(0, 100, (B, Lt))
    out  = m(ids, decoder_input_ids=dids, labels=labs)
    print(f"Loss: {out['loss'].item():.4f}   Logits: {out['logits'].shape}")
