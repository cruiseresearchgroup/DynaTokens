"""
llama_vqa.py
============
Builds the DynaTokens model on top of pretrained LLaMA weights.

build_model(args) -> DynaTokensTransformer
    1. reads `params.json` and the `*.pth` checkpoint(s) from args.llama_model_path
       (sharded checkpoints are gathered along their model-parallel dims);
    2. instantiates DynaTokensTransformer in fp16 and loads the backbone weights
       (strict=False: the DynaTokens modules are randomly initialised);
    3. freezes the backbone and unfreezes, in fp32, only the trainable parts:
         - token_generator      (H_phi)
         - z_task_current       (z_inv, z_sp)
         - temporal_emb, visual_proj
         - gate1 / gate2 of the last `adapter_layer` layers

Expected layout of args.llama_model_path (LLaMA-2-7B):
    params.json  tokenizer.model  consolidated.00.pth
"""

import json
from pathlib import Path

import torch

from llama import DynaTokensTransformer, ModelArgs, Tokenizer

TRAINABLE_MODULES = ('token_generator', 'z_task_current', 'temporal_emb', 'visual_proj')


def _load_llama_state_dict(model_path: str, n_layers: int) -> dict:
    """Load one or several LLaMA shards and merge them into a single state dict."""
    checkpoints = sorted(Path(model_path).glob("*.pth"))
    if len(checkpoints) == 0:
        raise RuntimeError(f"No .pth checkpoint found in {model_path}")

    loaded = []
    for ckpt in checkpoints:
        print(f"[llama_vqa] loading {ckpt}")
        loaded.append(torch.load(ckpt, map_location="cpu"))
    if len(loaded) == 1:
        return loaded[0]

    full_state_dict = {}

    def gather(name, dim):
        if dim < 0:                                   # replicated tensor
            full_state_dict[name] = loaded[0][name].clone()
        else:                                         # model-parallel tensor
            full_state_dict[name] = torch.cat([x[name] for x in loaded], dim=dim)
        for x in loaded:
            del x[name]

    gather("tok_embeddings.weight", 1)
    gather("norm.weight", -1)
    gather("output.weight", 0)
    for i in range(n_layers):
        prefix = f"layers.{i}."
        for key in ("attention_norm.weight", "ffn_norm.weight"):
            gather(prefix + key, -1)
        for key in ("attention.wq.weight", "attention.wk.weight", "attention.wv.weight",
                    "feed_forward.w1.weight", "feed_forward.w3.weight"):
            gather(prefix + key, 0)
        for key in ("attention.wo.weight", "feed_forward.w2.weight"):
            gather(prefix + key, 1)
    return full_state_dict


def build_model(args) -> DynaTokensTransformer:
    print(f'[llama_vqa] llama_model_path: {args.llama_model_path}')
    with open(f'{args.llama_model_path}/params.json', "r") as f:
        params = json.load(f)
    tokenizer = Tokenizer(model_path=f'{args.llama_model_path}/tokenizer.model')

    full_state_dict = _load_llama_state_dict(args.llama_model_path, params["n_layers"])

    model_args = ModelArgs(
        max_seq_len=args.max_seq_len,
        max_batch_size=32,
        adapter_len=args.adapter_len,
        adapter_layer=args.adapter_layer,
        **params,
    )
    model_args.vocab_size = tokenizer.n_words

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DynaTokensTransformer(model_args, args).to(device=device, dtype=torch.float16)

    missing, unexpected = model.load_state_dict(full_state_dict, strict=False)
    print(f"[llama_vqa] load_state_dict: {len(missing)} missing keys (DynaTokens modules), "
          f"{len(unexpected)} unexpected keys")

    # ── freeze backbone, unfreeze DynaTokens parts in fp32 ──
    for p in model.parameters():
        p.requires_grad = False

    adapter_start = len(model.layers) - model.adapter_layer
    for name, p in model.named_parameters():
        trainable = any(key in name for key in TRAINABLE_MODULES)
        if not trainable and ('gate1' in name or 'gate2' in name):
            parts = name.split('.')
            if len(parts) >= 2 and parts[0] == 'layers' and parts[1].isdigit():
                trainable = int(parts[1]) >= adapter_start
        if trainable:
            p.requires_grad = True
            if p.dtype == torch.float16:
                p.data = p.data.float()

    # ── summary ──
    stats = {}
    for name, p in model.named_parameters():
        if p.requires_grad:
            top = name.split('.')[0]
            stats.setdefault(top, [0, 0])
            stats[top][0] += 1
            stats[top][1] += p.numel()
    n_trainable = sum(v[1] for v in stats.values())
    n_total = sum(p.numel() for p in model.parameters())
    print(f"[llama_vqa] trainable parameters: {n_trainable:,} / {n_total:,}")
    for top, (cnt, numel) in sorted(stats.items()):
        print(f"    {top:22s} {cnt:4d} tensors  {numel:>12,} params")
    if 'token_generator' not in stats:
        raise RuntimeError("token_generator has no trainable parameters")
    return model
