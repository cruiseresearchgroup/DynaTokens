"""
llama/dynatokens_model.py
=========================
DynaTokens model: a frozen LLaMA backbone whose adapted layers receive prompt
tokens generated from a task code, plus the continual-learning machinery
(task codes, retrieval keys, task bank, routing).

Contents
--------
ModelArgs
    LLaMA hyper-parameters (read from params.json) plus the DynaTokens
    hyper-parameters (task-code size, token-generator size, bank size).
RMSNorm, precompute_freqs_cis, apply_rotary_emb, FeedForward
    Standard LLaMA building blocks (Meta, GPL-3 licensed).
Attention
    LLaMA attention with (i) prompt tokens prepended as extra keys/values,
    gated per head by `gate1` (zero-init, tanh), and (ii) a learnable
    video-to-text attention bias `gate2` (Flipped-VQA style).
TransformerBlock
    One LLaMA layer.
DynaTokensTransformer
    The full model. Main entry points:
      forward(data, inference=False, ...)   training losses / inference logits
      init_task / warm_start_z_sp_from_data / end_task
                                            task lifecycle for continual training
      push_current_to_bank_temp / pop_current_from_bank_temp
                                            temporarily add the task in progress to
                                            the bank so validation can route to it
      gen_all_layer_prompts_safe(z)         prompts for all adapted layers (OOM-safe)

Training forward returns a dict with 'vqa_loss', optional 'vaq_loss' /
'qav_loss' auxiliary losses (Flipped-VQA), and 'z_inv_reg' (Eq. 16).
Inference forward returns per-option negative log-likelihoods of shape
(B, n_options, seq_len); the option with the lowest mean NLL is the answer.
"""

import math
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from typing import Union

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import Embedding, Linear

from llama.dynatokens_components import (
    DecomposedTaskCode,
    RetrievalKeyModule,
    TaskBank,
    normalize_bank_key,
    warm_start_z_sp,
    z_inv_reg_loss,
)
from llama.token_generator import PromptEncoderTokenizer


@contextmanager
def train_mode(module: nn.Module, mode: bool):
    prev = module.training
    try:
        module.train(mode)
        yield
    finally:
        module.train(prev)


@dataclass
class ModelArgs:
    # ── LLaMA (overwritten from params.json) ──
    dim: int = 512
    n_layers: int = 8
    n_heads: int = 8
    vocab_size: int = -1
    multiple_of: int = 256
    norm_eps: float = 1e-5

    max_batch_size: int = 32
    max_seq_len: int = 2048
    adapter_len: int = 10          # N_p: prompt tokens per adapted layer
    adapter_layer: int = 30        # number of (last) layers that receive prompts

    # ── DynaTokens ──
    N_z: int = 30                  # number of task-code tokens
    C_z: int = 256                 # task-code channel dim
    internal_dim: int = 384        # token-generator width
    max_bank_size: int = 100       # max number of tasks stored in the bank
    max_len_generated_prompts: int = 32
    n_layers_token_generator: int = 2
    n_heads_token_generator: int = 8
    dropout_token_generator: float = 0.2

    # ── set from the training args (see DynaTokensTransformer.__init__) ──
    max_feats: int = 10            # number of video frames
    bias: float = 3.5              # initial video-attention bias (gate2 = -bias)


# ═══════════════════════════════════════════════════════════════════════════════
# LLaMA building blocks
# ═══════════════════════════════════════════════════════════════════════════════

class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        return self._norm(x.float()).type_as(x) * self.weight


def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)
    freqs = torch.outer(t, freqs).float()
    return torch.polar(torch.ones_like(freqs), freqs)


def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    ndim = x.ndim
    assert 0 <= 1 < ndim
    assert freqs_cis.shape == (x.shape[1], x.shape[-1])
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(*shape)


def apply_rotary_emb(xq, xk, freqs_cis):
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_)
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)


class Attention(nn.Module):
    """
    LLaMA attention with prompt tokens.

    `adapter` (B or 1, N_p, dim) is projected with wk/wv and prepended to the
    keys/values. The softmax over prompt positions is computed separately and
    scaled by tanh(gate1) (zero-initialised, so prompts have no effect at the
    start of training). `gate2` adds a learnable bias to text->video attention
    scores (initialised to -bias).
    """

    def __init__(self, args: ModelArgs):
        super().__init__()
        self.n_local_heads = args.n_heads
        self.head_dim = args.dim // args.n_heads
        self.max_feats = args.max_feats

        self.wq = Linear(args.dim, args.n_heads * self.head_dim, bias=False)
        self.wk = Linear(args.dim, args.n_heads * self.head_dim, bias=False)
        self.wv = Linear(args.dim, args.n_heads * self.head_dim, bias=False)
        self.wo = Linear(args.n_heads * self.head_dim, args.dim, bias=False)

        self.gate1 = torch.nn.Parameter(torch.zeros(1, self.n_local_heads, 1, 1))
        self.gate2 = torch.nn.Parameter(torch.ones(1, self.n_local_heads, 1, 1) * -args.bias)

    def forward(self, x, start_pos, freqs_cis, mask, adapter=None, video_start=None):
        bsz, seqlen, _ = x.shape
        device, dtype = x.device, x.dtype

        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)
        xq = xq.view(bsz, seqlen, self.n_local_heads, self.head_dim)
        xk = xk.view(bsz, seqlen, self.n_local_heads, self.head_dim)
        xv = xv.view(bsz, seqlen, self.n_local_heads, self.head_dim)
        xq, xk = apply_rotary_emb(xq, xk, freqs_cis=freqs_cis)

        if adapter is not None:
            adapter_len = adapter.shape[1]
            adapter = adapter.to(device=device, dtype=dtype)

            ak = self.wk(adapter)
            av = self.wv(adapter)
            if ak.dim() == 2:
                ak = ak.unsqueeze(0)
            if av.dim() == 2:
                av = av.unsqueeze(0)

            b_ad = ak.size(0)
            adapter_k = ak.view(b_ad, adapter_len, self.n_local_heads, self.head_dim)
            adapter_v = av.view(b_ad, adapter_len, self.n_local_heads, self.head_dim)
            if b_ad == 1 and bsz > 1:
                adapter_k = adapter_k.repeat(bsz, 1, 1, 1)
                adapter_v = adapter_v.repeat(bsz, 1, 1, 1)
            elif b_ad != bsz:
                raise RuntimeError(f"[Attention] adapter batch={b_ad} mismatch input batch={bsz}")

            gate1_expanded = self.gate1.to(device=device, dtype=dtype).expand(-1, -1, adapter_len, 1).half()
            adapter_k = adapter_k * (1 + gate1_expanded.transpose(1, 2))

            xk = torch.cat([adapter_k, xk], dim=1)
            xv = torch.cat([adapter_v, xv], dim=1)
            extra_mask = torch.zeros(1, 1, seqlen, adapter_len, device=device, dtype=dtype).to(mask)
            mask = torch.cat([extra_mask, mask], dim=-1)

        xq = xq.half().transpose(1, 2)
        keys = xk.half().transpose(1, 2)
        values = xv.half().transpose(1, 2)
        scores = torch.matmul(xq, keys.transpose(2, 3)) / math.sqrt(self.head_dim)
        if mask is not None:
            scores = scores + mask.to(device=device, dtype=scores.dtype)

        if adapter is not None:
            # softmax over prompt positions, scaled by tanh(gate1)
            adapter_scores = F.softmax(scores[..., :adapter_len].float(), dim=-1).type_as(xq)
            gate1_scale = gate1_expanded.transpose(1, 2).tanh().permute(0, 2, 1, 3).expand(bsz, -1, -1, -1)
            adapter_scores = adapter_scores * gate1_scale.transpose(-2, -1).half()

            # softmax over text/video positions with the video bias gate2
            if video_start is not None:
                vt_scores = scores[..., adapter_len:].clone()
                gate2_half = self.gate2.to(device=device, dtype=torch.float16)
                vs, ve = video_start, video_start + self.max_feats
                vt_scores[:, :, ve:, vs:ve] = vt_scores[:, :, ve:, vs:ve] + gate2_half
                vt_scores = F.softmax(vt_scores.float(), dim=-1).type_as(xq)
            else:
                vt_scores = F.softmax(scores[..., adapter_len:], dim=-1)
            scores = torch.cat([adapter_scores, vt_scores], dim=-1)
        else:
            scores = F.softmax(scores.float(), dim=-1).type_as(xq)

        output = torch.matmul(scores, values)
        output = output.transpose(1, 2).contiguous().view(bsz, seqlen, -1)
        return self.wo(output)


class FeedForward(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, multiple_of: int):
        super().__init__()
        hidden_dim = int(2 * hidden_dim / 3)
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)
        self.w1 = Linear(dim, hidden_dim, bias=False)
        self.w2 = Linear(hidden_dim, dim, bias=False)
        self.w3 = Linear(dim, hidden_dim, bias=False)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class TransformerBlock(nn.Module):
    def __init__(self, layer_id: int, args: ModelArgs):
        super().__init__()
        self.n_heads = args.n_heads
        self.dim = args.dim
        self.head_dim = args.dim // args.n_heads
        self.attention = Attention(args)
        self.feed_forward = FeedForward(dim=args.dim, hidden_dim=4 * args.dim, multiple_of=args.multiple_of)
        self.layer_id = layer_id
        self.attention_norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.ffn_norm = RMSNorm(args.dim, eps=args.norm_eps)

    def forward(self, x, start_pos, freqs_cis, mask, adapter=None, video_start=None):
        h = x + self.attention.forward(self.attention_norm(x), start_pos, freqs_cis, mask, adapter, video_start)
        return h + self.feed_forward.forward(self.ffn_norm(h))


class _TokenGenWrap(nn.Module):
    """Thin wrapper so `torch.func.functional_call` can swap generator parameters."""

    def __init__(self, token_generator: nn.Module):
        super().__init__()
        self.token_generator = token_generator

    def forward(self, z_t, layer_ids, z_pad_mask=None, n_prompts_keep=None):
        return self.token_generator.forward_for_layers(
            z_t=z_t, layer_ids=layer_ids, z_pad_mask=z_pad_mask, n_prompts_keep=n_prompts_keep,
        )


# ═══════════════════════════════════════════════════════════════════════════════
# DynaTokens
# ═══════════════════════════════════════════════════════════════════════════════

class DynaTokensTransformer(nn.Module):
    Key = Union[int, str]

    def __init__(self, params: ModelArgs, args):
        super().__init__()
        params.max_feats = args.max_feats
        params.bias = args.bias

        self.args = args
        self.params = params
        self.vocab_size = params.vocab_size
        self.n_layers = params.n_layers
        self.max_feats = args.max_feats
        self.adapter_len = params.adapter_len
        self.adapter_layer = params.adapter_layer

        # ── LLaMA backbone + Flipped-VQA video projection ──
        self.tok_embeddings = Embedding(params.vocab_size, params.dim)
        self.visual_proj = Linear(768, params.dim, bias=False)
        self.temporal_emb = Embedding(self.max_feats, params.dim)
        self.layers = torch.nn.ModuleList([TransformerBlock(i, params) for i in range(params.n_layers)])
        self.norm = RMSNorm(params.dim, eps=params.norm_eps)
        self.output = Linear(params.dim, params.vocab_size, bias=False)
        self.freqs_cis = precompute_freqs_cis(params.dim // params.n_heads, params.max_seq_len * 2)

        # ── losses ──
        self.vqa_criterion = torch.nn.CrossEntropyLoss(ignore_index=0)
        self.vaq_criterion = torch.nn.CrossEntropyLoss(ignore_index=0)
        self.qav_criterion = torch.nn.CrossEntropyLoss(ignore_index=-1)
        self.inference_criterion = torch.nn.CrossEntropyLoss(ignore_index=0, reduction='none')
        self.tau = args.tau

        # ── task code z^t = [z_inv, z_sp^t]  (Eq. 1) ──
        self.n_inv = params.N_z // 6
        self.n_sp = params.N_z - self.n_inv
        self.z_task_current = DecomposedTaskCode(n_inv=self.n_inv, n_sp=self.n_sp, c_z=params.C_z)
        print(f'[DynaTokens] N_z={params.N_z}, C_z={params.C_z}, internal_dim={params.internal_dim}, '
              f'n_inv={self.n_inv}, n_sp={self.n_sp}')

        # ── token generator H_phi  (Eq. 2) ──
        self.token_generator = PromptEncoderTokenizer(
            c_in=params.C_z,
            c_model=params.internal_dim,
            n_prompts=self.adapter_len,
            n_layers=params.n_layers_token_generator,
            n_heads=params.n_heads_token_generator,
            max_nz=params.max_len_generated_prompts,
            dropout=params.dropout_token_generator,
            n_target_layers=self.adapter_layer,
            c_out=params.dim,
        )
        self._tg_wrap = _TokenGenWrap(self.token_generator)

        # ── retrieval keys  (Eq. 8-11) ──
        self.key_mode = str(getattr(args, "key_mode", "raw"))
        self.d_k = int(getattr(args, "d_k", 256))
        self.retrieval_key_module = RetrievalKeyModule(
            d_llm=params.dim,
            d_k=self.d_k,
            beta=float(getattr(args, "key_ema_beta", 0.99)),
            mode=self.key_mode,
            d_video=768,
        )

        # ── task bank  (Eq. 13) ──
        self.task_bank = TaskBank(max_size=params.max_bank_size)
        self.task_bank.code_mode = str(getattr(args, "code_mode", "retrieved"))

        # ── z_inv regularisation  (Eq. 16) and warm-start temperature (Eq. 6) ──
        self.lambda_inv = float(getattr(args, "lambda_inv", 0.1))
        self.register_buffer("z_inv_snapshot", torch.zeros(self.n_inv, params.C_z, dtype=torch.float32))
        self.rho_z = float(getattr(args, "rho_z", 0.1))

        # ── internal state ──
        self._current_task_key = None
        self._temp_bank_key = None
        self._prompt_cache = {}
        self._prompt_cache_epoch = None

    # ═══════════════════════════════════════════════════════════════════════
    # Task lifecycle
    # ═══════════════════════════════════════════════════════════════════════

    def init_task(self, question_type=None, task_index: int = 1):
        """Start a new task: snapshot z_inv*, reset the key EMA and re-init z_sp."""
        self._current_task_key = self._make_task_key(question_type)
        self.z_inv_snapshot.copy_(self.z_task_current.snapshot_z_inv().to(self.z_inv_snapshot.device))
        self.retrieval_key_module.reset_ema()
        self.z_task_current.reset_z_sp(std=0.02)
        self._invalidate_prompt_cache()

    @torch.no_grad()
    def warm_start_z_sp_from_data(self):
        """Warm-start z_sp^t from the bank using the current EMA key k_0^t (Eq. 5-6)."""
        if len(self.task_bank) == 0:
            return
        k_current = self.retrieval_key_module.get_task_key()
        warm_start_z_sp(self.z_task_current, self.task_bank, k_current, rho_z=self.rho_z, noise_std=0.02)
        print(f"[DynaTokens] warm-started z_sp from {len(self.task_bank)} past tasks")

    def end_task(self, question_type=None):
        """Finish a task: push (k^t, z_sp^t) into the bank."""
        key = self._make_task_key(question_type or self._current_task_key)
        k_final = self.retrieval_key_module.get_task_key()
        self.task_bank.push(key, k_final, self.z_task_current.export_z_sp())
        print(f'[DynaTokens] end_task: {key}, bank size: {len(self.task_bank)}')
        self._invalidate_prompt_cache()
        self._current_task_key = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    @torch.no_grad()
    def push_current_to_bank_temp(self, question_type=None):
        """Temporarily add the task in progress to the bank (for mid-training validation)."""
        key = self._make_task_key(question_type or self._current_task_key)
        k = self.retrieval_key_module.get_task_key()
        self.task_bank.push(key, k, self.z_task_current.export_z_sp())
        self._temp_bank_key = key
        self._invalidate_prompt_cache()

    @torch.no_grad()
    def pop_current_from_bank_temp(self):
        """Remove the temporary entry added by push_current_to_bank_temp."""
        if self._temp_bank_key is not None:
            self.task_bank.remove(self._temp_bank_key)
        self._temp_bank_key = None
        self._invalidate_prompt_cache()

    # ═══════════════════════════════════════════════════════════════════════
    # Prompt generation
    # ═══════════════════════════════════════════════════════════════════════

    def gen_all_layer_prompts_safe(self, z_t, use_amp=True, amp_dtype=None, init_chunk_size=None):
        """
        Prompts for all adapted layers: (B, adapter_layer, N_p, dim).
        Tries a single batched call first and falls back to chunking over
        layers on CUDA OOM.
        """
        device = z_t.device
        L = self.adapter_layer
        layer_ids = torch.arange(L, device=device)
        amp_ctx = (
            torch.amp.autocast('cuda', dtype=(amp_dtype or torch.float16))
            if (use_amp and z_t.is_cuda) else nullcontext()
        )

        try:
            with amp_ctx:
                return self.token_generator.forward_for_layers(z_t=z_t, layer_ids=layer_ids)
        except RuntimeError as e:
            if "out of memory" not in str(e).lower():
                raise
            torch.cuda.empty_cache()

        chunk = init_chunk_size or max(1, L // 4)
        while chunk >= 1:
            try:
                pieces = []
                with amp_ctx:
                    for s in range(0, L, chunk):
                        pieces.append(self.token_generator.forward_for_layers(
                            z_t=z_t, layer_ids=layer_ids[s:s + chunk]))
                return torch.cat(pieces, dim=1)
            except RuntimeError as e:
                if "out of memory" not in str(e).lower():
                    raise
                torch.cuda.empty_cache()
                chunk //= 2
        raise RuntimeError("OOM in prompt generation")

    def _prompts_from_code(self, z_full, dev):
        """(1, N_z, C_z) -> fp16 prompts (1, L, N_p, dim)."""
        return self.gen_all_layer_prompts_safe(
            z_full.to(dev, dtype=torch.float32), use_amp=False,
        ).to(dtype=torch.float16, device=dev)

    # ═══════════════════════════════════════════════════════════════════════
    # Helpers
    # ═══════════════════════════════════════════════════════════════════════

    def _make_task_key(self, question_type):
        question_type = normalize_bank_key(question_type)
        if question_type is None:
            if not hasattr(self, "_task_counter"):
                self._task_counter = 0
            key = f"task_{self._task_counter:03d}"
            self._task_counter += 1
            return key
        return question_type

    def _invalidate_prompt_cache(self):
        self._prompt_cache.clear()
        self._prompt_cache_epoch = None

    def _maybe_invalidate_prompt_cache_for_eval(self, inference, current_epoch=None):
        if not inference:
            return
        if current_epoch is None or self._prompt_cache_epoch != current_epoch:
            self._invalidate_prompt_cache()
            self._prompt_cache_epoch = current_epoch

    def _get_single_task_key_from_batch(self, data, question_type=None):
        """Ground-truth task key of a (single-task) batch, from data['task_key'] or the argument."""
        gt_task_key = None
        if isinstance(data, dict) and "task_key" in data:
            tk = data["task_key"]
            if isinstance(tk, (list, tuple)):
                uniq = list(dict.fromkeys(normalize_bank_key(x) for x in tk))
                assert len(uniq) == 1, f"batch mixes several tasks: {uniq}"
                gt_task_key = uniq[0]
            else:
                gt_task_key = normalize_bank_key(tk)
        if gt_task_key is None and question_type is not None:
            gt_task_key = normalize_bank_key(question_type)
        return gt_task_key

    def extract_question_from_flat(self, ids, label_full, video_start, max_feats):
        """
        Indices of the question tokens in a flat VQA sequence: everything
        between the end of the video block and the first answer label.
        Returns (q_idx (B, Lq), q_pad_mask (B, Lq), q_start).
        """
        B, S = ids.shape
        device = ids.device
        pos = (label_full > 0)
        has = pos.any(dim=1)
        prefix = torch.full((B,), S, device=device, dtype=torch.long)
        prefix[has] = pos[has].to(torch.int32).argmax(dim=1)
        q_start = min(int(video_start + max_feats + 1), S)
        q_len = (prefix - q_start).clamp(min=0, max=S - q_start)
        Lq = int(q_len.max().item()) if B > 0 else 0
        if Lq == 0:
            return (torch.full((B, 1), q_start, device=device, dtype=torch.long),
                    torch.zeros(B, 1, dtype=torch.bool, device=device), q_start)
        ar = torch.arange(Lq, device=device).unsqueeze(0)
        q_idx = (q_start + ar).expand(B, -1)
        end = (prefix - 1).clamp(min=0).unsqueeze(1)
        q_idx = torch.minimum(q_idx, end)
        q_pad_mask = (ar < q_len.unsqueeze(1))
        return q_idx, q_pad_mask, q_start

    def _compute_features_for_keys(self, video, vqa_id, vqa_label, vqa_video_start):
        """
        Frozen features for retrieval keys: question token embeddings from
        the (frozen) LLaMA embedding table and the raw video features.
        Returns (question_tokens (B, Lq, dim), raw_video (B, max_feats, 768)).
        """
        route_vqa_id = vqa_id[:, 0, :]
        route_vqa_label = vqa_label[:, 0, :]
        route_q_idx, _, _ = self.extract_question_from_flat(
            route_vqa_id, route_vqa_label, vqa_video_start, self.max_feats)
        with torch.no_grad():
            vqa_h = self.tok_embeddings(route_vqa_id).float()
        question_tokens = vqa_h.gather(
            dim=1, index=route_q_idx.unsqueeze(-1).expand(-1, -1, vqa_h.size(-1))).float()
        return question_tokens, video.float()

    # ═══════════════════════════════════════════════════════════════════════
    # Inference routing  (Eq. 14-15)
    # ═══════════════════════════════════════════════════════════════════════

    @torch.no_grad()
    def _retrieve_and_generate_prompts(self, sample_keys, B, n_options, *, dev,
                                       use_task_id=False, data=None, question_type=None,
                                       return_routing_stats=False):
        """
        Pick a task code for every sample and generate its prompts.

        * empty bank      -> current task code
        * use_task_id     -> oracle: look the ground-truth task up in the bank
        * otherwise       -> cosine retrieval with `sample_keys` (B, D_k)

        Returns (P_all (B * n_options, L, N_p, dim), routing_stats or None).
        """
        routing_stats = None

        if len(self.task_bank) == 0:
            P_all = self._prompts_from_code(self.z_task_current.full_code_batch(), dev)
            return P_all.expand(B, -1, -1, -1).repeat_interleave(n_options, dim=0), routing_stats

        if use_task_id:
            task_key = normalize_bank_key(self._get_single_task_key_from_batch(data, question_type))
            entry = self.task_bank.bank.get(task_key)
            if entry is None:
                z_full = self.z_task_current.full_code_batch()
            else:
                z_full = torch.cat([self.z_task_current.z_inv.detach().to(dev),
                                    entry["z_sp"].to(dev)], dim=0).unsqueeze(0)
            P_all = self._prompts_from_code(z_full, dev)
            return P_all.expand(B, -1, -1, -1).repeat_interleave(n_options, dim=0), routing_stats

        z_inv = self.z_task_current.z_inv.detach()
        pred_task_ids, _ = self.task_bank.retrieve_batch(sample_keys, z_inv)

        if return_routing_stats:
            gt_key = normalize_bank_key(self._get_single_task_key_from_batch(data, question_type))
            correct = sum(int(pk == gt_key) for pk in pred_task_ids)
            route_confusion = {}
            for pk in pred_task_ids:
                route_confusion[pk] = route_confusion.get(pk, 0) + 1
            routing_stats = {
                "route_correct": correct,
                "route_total": len(pred_task_ids),
                "route_acc": correct / max(len(pred_task_ids), 1),
                "route_gt_key": gt_key,
                "route_pred_keys": pred_task_ids,
                "route_confusion": route_confusion,
            }

        # ── code-swap ablation (routing analysis in the paper) ──
        #   'random'    : real z_inv + random z_sp (norm matched to a real z_sp)
        #   'zsp_zero'  : real z_inv + z_sp = 0     (z_inv alone)
        #   'zinv_zero' : z_inv = 0   + retrieved z_sp (z_sp alone)
        mode = self.task_bank.code_mode
        if mode in ('random', 'zsp_zero', 'zinv_zero'):
            z_inv_real = z_inv.to(dev)
            ref_z_sp = next(iter(self.task_bank.bank.values()))["z_sp"].to(dev)
            if mode == 'random':
                z_sp_use = torch.randn_like(ref_z_sp)
                z_sp_use = z_sp_use * (ref_z_sp.norm() / z_sp_use.norm())
                z_inv_use = z_inv_real
            elif mode == 'zsp_zero':
                z_sp_use = torch.zeros_like(ref_z_sp)
                z_inv_use = z_inv_real
            else:
                z_sp_use = self.task_bank.bank[pred_task_ids[0]]["z_sp"].to(dev)
                z_inv_use = torch.zeros_like(z_inv_real)
            P_ab = self._prompts_from_code(torch.cat([z_inv_use, z_sp_use], dim=0).unsqueeze(0), dev)
            return P_ab.expand(B, -1, -1, -1).repeat_interleave(n_options, dim=0), routing_stats

        # prompts per retrieved task, cached by task key
        for k in dict.fromkeys(pred_task_ids):
            if k not in self._prompt_cache:
                z_k = torch.cat([z_inv.to(dev), self.task_bank.bank[k]["z_sp"].to(dev)], dim=0).unsqueeze(0)
                self._prompt_cache[k] = self._prompts_from_code(z_k, dev)

        P_list = [self._prompt_cache[k] for k in pred_task_ids]
        P_route = torch.cat(P_list, dim=0) if P_list[0].size(0) == 1 else torch.stack(P_list)
        P_all = P_route.to(dtype=torch.float16, device=dev).repeat_interleave(n_options, dim=0)
        return P_all, routing_stats

    # ═══════════════════════════════════════════════════════════════════════
    # Forward
    # ═══════════════════════════════════════════════════════════════════════

    def forward(self, data, inference=False, current_epoch=None, question_type=None,
                hp_override=None, return_routing_stats=False):
        """
        data : batch dict from dataloader.batch_collate
        inference : False -> dict of losses; True -> per-option NLL (B, n_options, S-1)
        hp_override : {'P_all': prompts} to bypass prompt generation (LookAhead inner loop)
        return_routing_stats : also return routing statistics at inference
        """
        dev = next(self.parameters()).device
        dtype = next(self.parameters()).dtype
        self._maybe_invalidate_prompt_cache_for_eval(inference, current_epoch)
        use_vaq = getattr(self.args, 'vaq', False) and not inference
        use_qav = getattr(self.args, 'qav', False) and not inference

        # ── unpack ──
        video = data['video'].to(dev, non_blocking=True)
        vqa_id = data['text_id']['vqa'].to(dev, non_blocking=True)
        vqa_label = data['label']['vqa'].to(dev, non_blocking=True)
        vqa_video_start = int(data['video_start']['vqa'][0])

        vaq_id = data['text_id']['vaq'].to(dev, non_blocking=True)
        vaq_label = data['label']['vaq'].to(dev, non_blocking=True)
        vaq_video_start = int(data['video_start']['vaq'][0])

        qav_id = data['text_id']['qav'].to(dev, non_blocking=True)
        qav_label = data['label']['qav'].to(dev, non_blocking=True)
        qav_video_index = data['video_index']['qav'].to(dev, non_blocking=True)

        bsz, n_options, seqlen = vqa_id.shape
        ids_flat = vqa_id.reshape(-1, seqlen)
        vqa_label_full = vqa_label.reshape(-1, seqlen)
        vqa_label_for_ce = vqa_label_full[:, 1:].flatten()

        vaq_id = vaq_id.reshape(-1, seqlen)
        vaq_label = vaq_label.reshape(-1, seqlen)[:, 1:].flatten()

        qav_id = qav_id.reshape(-1, seqlen)
        qav_label = qav_label.reshape(-1, seqlen)
        qav_video_mask = qav_label.ge(0)
        qav_label = qav_label[:, 1:].flatten()

        # ── token embeddings (frozen) ──
        with torch.no_grad():
            vqa_h = self.tok_embeddings(ids_flat).to(dtype=dtype)
            if use_vaq:
                vaq_h = self.tok_embeddings(vaq_id).to(dtype=dtype)
            if use_qav:
                qav_h = self.tok_embeddings(qav_id).to(dtype=dtype)

        freqs_cis = self.freqs_cis.to(vqa_h.device)[:seqlen]
        mask = torch.full((1, 1, seqlen, seqlen), float("-inf"), device=dev, dtype=dtype)
        mask = torch.triu(mask, diagonal=1)

        # ── video features, inserted in place of the video placeholder tokens ──
        _vf = self.visual_proj(video).to(dtype=dtype)
        if inference:
            _vf = _vf.unsqueeze(1).repeat(1, n_options, 1, 1).view(-1, _vf.shape[-2], _vf.shape[-1])
        temb = self.temporal_emb.weight.to(device=dev, dtype=dtype)
        max_fp16 = torch.finfo(torch.float16).max
        video_feature = torch.clamp(_vf + temb[None, :, :], -max_fp16, max_fp16).to(torch.float16)

        vqa_h = vqa_h.clone()
        vqa_h[:, vqa_video_start:vqa_video_start + self.max_feats] = video_feature
        if use_vaq:
            vaq_h = vaq_h.clone()
            vaq_h[:, vaq_video_start:vaq_video_start + self.max_feats] = video_feature
        if use_qav:
            qav_h = qav_h * ~qav_video_mask[..., None]
            qav_h.scatter_add_(1, qav_video_index[..., None].repeat(1, 1, self.params.dim), video_feature)

        # ── retrieval key EMA from frozen features (training only) ──
        if not inference:
            with torch.no_grad():
                q_feat, v_feat = self._compute_features_for_keys(video, vqa_id, vqa_label, vqa_video_start)
                self.retrieval_key_module.update_ema(
                    self.retrieval_key_module.compute_sample_keys(q_feat, v_feat))

        # ── prompt generation ──
        routing_stats = None
        if hp_override is not None and 'P_all' in hp_override:
            P_all = hp_override['P_all'].to(dtype=torch.float16, device=dev)
        elif inference:
            with torch.no_grad():
                q_feat, v_feat = self._compute_features_for_keys(video, vqa_id, vqa_label, vqa_video_start)
                sample_keys = self.retrieval_key_module.compute_sample_keys(q_feat, v_feat)
                P_all, routing_stats = self._retrieve_and_generate_prompts(
                    sample_keys, q_feat.shape[0], n_options, dev=dev,
                    use_task_id=getattr(self.args, "use_task_id", False),
                    data=data, question_type=question_type,
                    return_routing_stats=return_routing_stats,
                )
        else:
            P_all = self._prompts_from_code(self.z_task_current.full_code_batch(), dev)

        P_all = torch.clamp(P_all, -max_fp16, max_fp16).to(torch.float16)

        # ── LLaMA forward; prompts go to the last `adapter_layer` layers ──
        start_pos = 0
        for i, layer in enumerate(self.layers[-self.adapter_layer:]):
            P_i = P_all[:, i, :, :].to(device=dev)
            vqa_h = layer(vqa_h, start_pos, freqs_cis, mask, P_i, vqa_video_start)
            if use_vaq:
                vaq_h = layer(vaq_h, start_pos, freqs_cis, mask, P_i, vaq_video_start)
            if use_qav:
                qav_h = layer(qav_h, start_pos, freqs_cis, mask, P_i, None)

        # ── losses ──
        vqa_output = self.output(self.norm(vqa_h))[:, :-1, :].reshape(-1, self.vocab_size)
        vqa_loss = self.vqa_criterion(vqa_output, vqa_label_for_ce)

        if inference:
            logits = self.inference_criterion(vqa_output, vqa_label_for_ce).reshape(bsz, n_options, -1)
            return (logits, routing_stats) if return_routing_stats else logits

        losses = {
            'vqa_loss': vqa_loss,
            'vaq_loss': torch.tensor(0.0, device=dev),
            'qav_loss': torch.tensor(0.0, device=dev),
            'z_inv_reg': z_inv_reg_loss(self.z_task_current.z_inv, self.z_inv_snapshot),
            'lambda_inv': self.lambda_inv,
        }
        if use_vaq:
            vaq_output = self.output(self.norm(vaq_h))[:, :-1, :].reshape(-1, self.vocab_size)
            losses['vaq_loss'] = self.vaq_criterion(vaq_output, vaq_label)
        if use_qav:
            qav_h = self.norm(qav_h)
            video_key = video_feature
            if video_key.shape[0] != qav_h.shape[0]:
                video_key = video_key.repeat_interleave(qav_h.shape[0] // video_key.shape[0], dim=0)
            qav_output = torch.bmm(qav_h[:, :-1].float(), video_key.transpose(1, 2).float()).reshape(-1, self.max_feats)
            losses['qav_loss'] = self.qav_criterion(qav_output / self.tau, qav_label)
        return losses
