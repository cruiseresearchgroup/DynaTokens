"""
llama/token_generator.py
========================
The token generator H_phi (Eq. 2): a small transformer that maps a task code
z^t in R^{N_z x C_z} to a set of N_p prompt tokens for every adapted LLaMA
layer. The generated tokens are prepended as extra keys/values in the
attention of each adapted layer (see llama/dynatokens_model.py).

Classes
-------
PositionEmbedding1D
    Learned positional embedding over the N_z code tokens (interpolated if
    the code is longer than max_len).
FFN
    Pre-norm feed-forward block (GELU, 2x expansion).
CrossSelfBlock
    One generator block: cross-attention (prompt queries -> code tokens),
    self-attention over the prompt queries, FFN.
LowRankPromptQueryGenerator
    Produces the initial prompt queries Z^{q,t} from the code by attention
    pooling followed by a low-rank expansion.
PromptEncoderTokenizer
    H_phi itself. `forward` generates prompts for one target layer per batch
    element; `forward_for_layers` vectorises over all target layers and
    returns (B, L, N_p, c_out). Layer conditioning is done with FiLM from a
    learned layer embedding.
"""

from typing import Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionEmbedding1D(nn.Module):
    """Learned 1-D positional embedding, linearly interpolated beyond max_len."""

    def __init__(self, dim: int, max_len: int = 32):
        super().__init__()
        self.dim = dim
        self.max_len = max_len
        self.pe = nn.Parameter(torch.zeros(1, max_len, dim))
        nn.init.trunc_normal_(self.pe, std=0.02)

    def forward(self, x: torch.Tensor, seq_len: Optional[int] = None) -> torch.Tensor:
        """x: (B, L, C) -> (1, L_req, C)"""
        _, L, _ = x.shape
        L_req = seq_len or L
        if L_req == self.max_len:
            return self.pe
        if L_req < self.max_len:
            return self.pe[:, :L_req, :]
        pe = self.pe.transpose(1, 2)                                  # (1, C, max_len)
        pe = F.interpolate(pe, size=L_req, mode="linear", align_corners=False)
        return pe.transpose(1, 2)                                     # (1, L_req, C)


class FFN(nn.Module):
    """Feed-forward block with 2x hidden expansion (lighter than the usual 4x)."""

    def __init__(self, dim: int, hidden_dim: Optional[int] = None, dropout: float = 0.0):
        super().__init__()
        hidden_dim = hidden_dim or (2 * dim)
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )
        self.apply(_init_linear)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def _init_linear(m: nn.Module):
    if isinstance(m, nn.Linear):
        nn.init.trunc_normal_(m.weight, std=0.02)
        if m.bias is not None:
            nn.init.zeros_(m.bias)


class CrossSelfBlock(nn.Module):
    """Pre-norm block: cross-attention -> self-attention -> FFN."""

    def __init__(self, dim: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        self.ln_q1 = nn.LayerNorm(dim)
        self.cross_attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.drop1 = nn.Dropout(dropout)

        self.ln_q2 = nn.LayerNorm(dim)
        self.self_attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.drop2 = nn.Dropout(dropout)

        self.ln_mlp = nn.LayerNorm(dim)
        self.ffn = FFN(dim, dropout=dropout)

    def forward(
        self,
        U: torch.Tensor,
        Zmem: torch.Tensor,
        mem_pad_mask: Optional[torch.Tensor] = None,
        attn_mask_self: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        U1 = self.ln_q1(U)
        U_cross, _ = self.cross_attn(query=U1, key=Zmem, value=Zmem, key_padding_mask=mem_pad_mask)
        U = U + self.drop1(U_cross)

        U2 = self.ln_q2(U)
        U_self, _ = self.self_attn(query=U2, key=U2, value=U2, attn_mask=attn_mask_self)
        U = U + self.drop2(U_self)

        U = U + self.ffn(self.ln_mlp(U))
        return U


class LowRankPromptQueryGenerator(nn.Module):
    """
    Initial prompt queries from the code tokens:

      Zt (B, Nz, C) --attention pooling per prompt--> pooled (B, Np, C)
                    --MLP--> (B, Np, r) --basis (r, C)--> (B, Np, C) + prompt_pos
    """

    def __init__(self, dim: int, n_prompts: int, hidden: Optional[int] = None,
                 rank: int = 32, dropout: float = 0.0):
        super().__init__()
        hidden = hidden or (2 * dim)
        self.n_prompts = n_prompts
        self.rank = int(rank)

        self.proj_att = nn.Linear(dim, n_prompts)                    # attention logits per prompt
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, self.rank),
        )
        self.basis = nn.Parameter(torch.empty(self.rank, dim))       # shared low-rank basis
        nn.init.trunc_normal_(self.basis, std=0.02)
        self.prompt_pos = nn.Parameter(torch.zeros(1, n_prompts, dim))
        nn.init.trunc_normal_(self.prompt_pos, std=0.02)

        self.apply(_init_linear)

    def forward(self, Zt: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Zt: (B, Nz, C); mask: (B, Nz) with True = pad. Returns (B, Np, C)."""
        B, Nz, C = Zt.shape
        att_logits = self.proj_att(Zt)                               # (B, Nz, Np)
        if mask is not None:
            att_logits = att_logits.masked_fill(mask.unsqueeze(-1), float("-inf"))
        att = F.softmax(att_logits, dim=1)                           # softmax over code tokens
        pooled = torch.einsum("bnc,bnp->bpc", Zt, att)               # (B, Np, C)

        qr = self.mlp(pooled.contiguous().view(B * self.n_prompts, C))
        qr = qr.view(B, self.n_prompts, self.rank)                   # (B, Np, r)
        return torch.einsum("bnr,rc->bnc", qr, self.basis) + self.prompt_pos


class PromptEncoderTokenizer(nn.Module):
    """
    H_phi: task code -> per-layer prompt tokens.

    Args
    ----
    c_in            : C_z, channel dim of the task code
    c_model         : internal width of the generator
    n_prompts       : N_p, number of prompt tokens per layer (= adapter_len)
    n_layers        : number of CrossSelfBlocks
    n_heads         : attention heads
    max_nz          : max number of code tokens for the positional embedding
    dropout         : dropout inside the generator
    n_target_layers : number of LLaMA layers that receive prompts (= adapter_layer)
    q_rank          : rank of the low-rank query head
    c_out           : output dim (= LLaMA hidden dim, 4096 for 7B)
    """

    def __init__(
        self,
        c_in: int,
        c_model: int,
        n_prompts: int,
        n_layers: int = 4,
        n_heads: int = 8,
        max_nz: int = 32,
        dropout: float = 0.2,
        n_target_layers: int = 32,
        q_rank: int = 32,
        c_out: Optional[int] = None,
    ):
        super().__init__()
        assert c_model % n_heads == 0, "c_model must be divisible by n_heads"

        self.c_model = c_model
        self.c_out = c_out if c_out is not None else c_model
        self.n_prompts = n_prompts
        self.n_layers = n_layers
        self.n_target_layers = int(n_target_layers)

        # (1) project code tokens + positional embedding
        self.proj_z = nn.Linear(c_in, c_model, bias=True)
        nn.init.trunc_normal_(self.proj_z.weight, std=0.02)
        nn.init.zeros_(self.proj_z.bias)
        self.pos_z = PositionEmbedding1D(c_model, max_len=max_nz)

        # (2) low-rank prompt queries
        self.query_gen = LowRankPromptQueryGenerator(
            dim=c_model, n_prompts=n_prompts, hidden=2 * c_model, rank=q_rank, dropout=dropout,
        )

        # (3) U^(0) = Z^{q,t} + MHCA(LN(Z^{q,t}), Z^t, Z^t)
        self.init_ln = nn.LayerNorm(c_model)
        self.init_cross = nn.MultiheadAttention(c_model, n_heads, dropout=dropout, batch_first=True)
        self.init_drop = nn.Dropout(dropout)

        # (4) generator blocks
        self.blocks = nn.ModuleList(
            [CrossSelfBlock(dim=c_model, num_heads=n_heads, dropout=dropout) for _ in range(n_layers)]
        )

        # (5) output head: internal width -> LLaMA hidden dim
        self.W_o = nn.Linear(c_model, self.c_out)
        nn.init.trunc_normal_(self.W_o.weight, std=0.02)
        nn.init.zeros_(self.W_o.bias)

        # (6) layer conditioning (FiLM) from a learned layer embedding
        self.layer_table = nn.Embedding(self.n_target_layers, c_model)
        nn.init.trunc_normal_(self.layer_table.weight, std=0.02)
        self.cond_proj = nn.Sequential(nn.LayerNorm(c_model), nn.Linear(c_model, 2 * c_model))
        nn.init.trunc_normal_(self.cond_proj[1].weight, std=0.02)
        nn.init.zeros_(self.cond_proj[1].bias)

    # ── helpers ──

    def _apply_film(self, X: torch.Tensor, layer_emb: torch.Tensor) -> torch.Tensor:
        gamma, beta = self.cond_proj(layer_emb).chunk(2, dim=-1)     # (B, C) each
        gamma = 1.0 + torch.tanh(gamma)
        return X * gamma.unsqueeze(1) + beta.unsqueeze(1)

    @staticmethod
    def _normalize_layer_idx(layer_idx: Union[int, torch.LongTensor], B: int, device) -> torch.LongTensor:
        if isinstance(layer_idx, int):
            idx = torch.tensor([layer_idx], device=device, dtype=torch.long)
        else:
            idx = layer_idx.to(device=device, dtype=torch.long)
        if idx.ndim == 0:
            idx = idx.view(1)
        if idx.numel() == 1 and B > 1:
            idx = idx.expand(B)
        if idx.shape[0] != B:
            raise ValueError(f"layer_idx batch mismatch: got {idx.shape[0]} vs B={B}")
        return idx

    # ── forward ──

    def forward(
        self,
        z_t: torch.Tensor,
        z_pad_mask: Optional[torch.Tensor] = None,
        n_prompts_keep: Optional[int] = None,
        layer_idx: Optional[Union[int, torch.LongTensor]] = None,
    ) -> torch.Tensor:
        """
        z_t: (B, Nz, C_z); layer_idx: target layer per batch element.
        Returns (B, n_prompts_keep or Np, c_out).
        """
        B, Nz, _ = z_t.shape
        dev = z_t.device

        Zt = self.proj_z(z_t)
        Zt = Zt + self.pos_z(Zt, seq_len=Nz)

        layer_emb = None
        if layer_idx is not None:
            layer_emb = self.layer_table(self._normalize_layer_idx(layer_idx, B=B, device=dev))
            Zt = self._apply_film(Zt, layer_emb)

        Z_qt = self.query_gen(Zt, mask=z_pad_mask)
        if layer_emb is not None:
            Z_qt = self._apply_film(Z_qt, layer_emb)

        U0 = self.init_ln(Z_qt)
        U_cross, _ = self.init_cross(query=U0, key=Zt, value=Zt, key_padding_mask=z_pad_mask)
        U = Z_qt + self.init_drop(U_cross)

        for blk in self.blocks:
            U = blk(U, Zt, mem_pad_mask=z_pad_mask)

        if n_prompts_keep is not None:
            U = U[:, :n_prompts_keep, :]
        return self.W_o(U)

    def forward_for_layers(
        self,
        z_t: torch.Tensor,
        layer_ids: Optional[torch.LongTensor] = None,
        z_pad_mask: Optional[torch.Tensor] = None,
        n_prompts_keep: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Vectorised generation for several target layers.
        z_t: (B, Nz, C_z); layer_ids: (L,) or (B, L).
        Returns (B, L, n_prompts_keep or Np, c_out).
        """
        B, Nz, Cz = z_t.shape
        if layer_ids is None:
            layer_ids = torch.arange(self.n_target_layers, device=z_t.device)
        if layer_ids.ndim == 1:
            layer_ids = layer_ids.view(1, -1).expand(B, -1)
        elif layer_ids.ndim == 2 and layer_ids.shape[0] == 1 and B > 1:
            layer_ids = layer_ids.expand(B, -1)
        assert layer_ids.shape[0] == B
        L = layer_ids.shape[1]

        z_b = z_t.unsqueeze(1).expand(B, L, Nz, Cz).reshape(B * L, Nz, Cz)
        m_b = (
            z_pad_mask.unsqueeze(1).expand(B, L, Nz).reshape(B * L, Nz)
            if z_pad_mask is not None else None
        )
        P_flat = self.forward(z_b, z_pad_mask=m_b, n_prompts_keep=n_prompts_keep,
                              layer_idx=layer_ids.reshape(B * L))
        return P_flat.view(B, L, P_flat.size(1), P_flat.size(2))
