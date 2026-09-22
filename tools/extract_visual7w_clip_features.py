#!/usr/bin/env python3
"""
tools/extract_visual7w_clip_features.py
=======================================
Extract one CLIP ViT-L/14 (OpenAI weights, via open_clip) image embedding per
Visual7W image and save them in the format read by dataloader/visual7w.py:

    <out_dir>/clipvitl14.pth = {
        "features":  {image_id (int): tensor(768)},
        "filenames": {image_id (int): "v7w_<id>.jpg"},
        "meta":      {...},
    }

Usage
-----
    pip install open_clip_torch pillow tqdm

    # all images referenced by the provided splits, single process
    python tools/extract_visual7w_clip_features.py \\
        --dataset_json data/visual7w/train.json data/visual7w/val.json \\
        --image_roots /path/to/visual7w/images \\
        --out_dir data/visual7w

    # split the work over N jobs (e.g. one per GPU), then merge
    python tools/extract_visual7w_clip_features.py ... --num_shards 4 --shard_id 0   # ... 1, 2, 3
    python tools/extract_visual7w_clip_features.py --merge --out_dir data/visual7w

Images are located as <root>/<filename> or <root>/<split>/<filename> for every
root in --image_roots. Features are stored un-normalised in fp32 unless
--normalize / --fp16 are given.
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from torch.utils.data import DataLoader, Dataset

DICT_NAME = "clipvitl14.pth"


# ═══════════════════════════════════════════════════════════════════════════════
# Records and image paths
# ═══════════════════════════════════════════════════════════════════════════════

def load_image_records(json_paths: List[str]) -> List[Dict]:
    """Image records ({"image_id", "filename", "split", ...}) of one or more split files."""
    records = []
    for p in json_paths:
        with open(p, "r") as f:
            data = json.load(f)
        if not isinstance(data.get("images"), list):
            raise ValueError(f"{p}: expected a top-level list 'images'")
        records.extend(data["images"])
    return records


def resolve_image_path(rec: Dict, image_roots: List[str]) -> str:
    """<root>/<filename> or <root>/<split>/<filename>, first match wins."""
    fname = rec.get("filename")
    if not fname:
        raise ValueError(f"Missing 'filename' in record with keys {list(rec.keys())}")
    split = rec.get("split")
    candidates = []
    for root in image_roots:
        candidates.append(Path(root) / fname)
        if split:
            candidates.append(Path(root) / split / fname)
    for p in candidates:
        if p.exists():
            return str(p)
    raise FileNotFoundError(
        f"Cannot find '{fname}' (split={split}). Tried: " + ", ".join(str(c) for c in candidates)
        + ". Set --image_roots to the folder containing the images (or its parent with train/val/test)."
    )


class ImagePathDataset(Dataset):
    """(image_id, path, filename) -> (image_id, filename, preprocessed image)."""

    def __init__(self, items: List[Tuple[int, str, str]], preprocess):
        self.items = items
        self.preprocess = preprocess

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        from PIL import Image
        image_id, path, filename = self.items[idx]
        return image_id, filename, self.preprocess(Image.open(path).convert("RGB"))


# ═══════════════════════════════════════════════════════════════════════════════
# Extraction / merge
# ═══════════════════════════════════════════════════════════════════════════════

def shard_path(out_dir: Path, shard_id: int, num_shards: int) -> Path:
    return out_dir / f"clipvitl14_shard{shard_id:03d}_of_{num_shards:03d}.pth"


@torch.no_grad()
def extract(args):
    import open_clip
    from tqdm import tqdm

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    records = load_image_records(args.dataset_json)
    if args.splits:
        records = [r for r in records if r.get("split") in set(args.splits)]

    items, seen = [], set()
    for rec in tqdm(records, desc="Resolving image paths"):
        image_id = int(rec["image_id"])
        if image_id in seen:
            continue
        seen.add(image_id)
        items.append((image_id, resolve_image_path(rec, args.image_roots), rec.get("filename", "")))
    items.sort(key=lambda x: x[0])
    if not (0 <= args.shard_id < args.num_shards):
        raise ValueError("shard_id must be in [0, num_shards)")
    items = items[args.shard_id::args.num_shards]
    print(f"[extract] {len(seen)} unique images, shard {args.shard_id}/{args.num_shards}: {len(items)} images")
    if not items:
        raise RuntimeError("No images to process; check --image_roots / --dataset_json")

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model, _, preprocess = open_clip.create_model_and_transforms("ViT-L-14", pretrained="openai")
    model = model.to(device).eval()

    loader = DataLoader(ImagePathDataset(items, preprocess), batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, pin_memory=True)

    features: Dict[int, torch.Tensor] = {}
    filenames: Dict[int, str] = {}
    for image_ids, fnames, x in tqdm(loader, desc="Extracting CLIP features"):
        feats = model.encode_image(x.to(device, non_blocking=True)).float()
        if args.normalize:
            feats = feats / feats.norm(dim=-1, keepdim=True).clamp(min=1e-12)
        feats = feats.cpu()
        if args.fp16:
            feats = feats.half()
        for i in range(feats.shape[0]):
            iid = int(image_ids[i])
            features[iid] = feats[i].contiguous()
            filenames[iid] = fnames[i]

    meta = {
        "model": "CLIP ViT-L/14 (openai, open_clip)",
        "feature_dim": int(next(iter(features.values())).shape[-1]),
        "num_images": len(features),
        "fp16": bool(args.fp16),
        "normalized": bool(args.normalize),
        "shard_id": args.shard_id,
        "num_shards": args.num_shards,
    }
    path = out_dir / DICT_NAME if args.num_shards == 1 else shard_path(out_dir, args.shard_id, args.num_shards)
    torch.save({"features": features, "filenames": filenames, "meta": meta}, path)
    print(f"[extract] saved {len(features)} features ({meta['feature_dim']}-d) -> {path}")


def merge(args):
    """Merge clipvitl14_shard*.pth files in out_dir into clipvitl14.pth."""
    out_dir = Path(args.out_dir)
    shards = sorted(out_dir.glob("clipvitl14_shard*_of_*.pth"))
    if not shards:
        raise FileNotFoundError(f"No shard files found in {out_dir}")
    features, filenames, meta = {}, {}, None
    for p in shards:
        d = torch.load(p, map_location="cpu", weights_only=False)
        features.update(d["features"])
        filenames.update(d.get("filenames", {}))
        meta = meta or d.get("meta", {})
        print(f"[merge] {p.name}: {len(d['features'])} features")
    meta = dict(meta or {}, num_images=len(features), shard_id=None, num_shards=len(shards))
    torch.save({"features": features, "filenames": filenames, "meta": meta}, out_dir / DICT_NAME)
    print(f"[merge] saved {len(features)} features -> {out_dir / DICT_NAME}")


def get_args():
    ap = argparse.ArgumentParser(description="CLIP ViT-L/14 image features for Visual7W")
    ap.add_argument("--dataset_json", type=str, nargs="+", default=["data/visual7w/train.json", "data/visual7w/val.json"],
                    help="split file(s) listing the images")
    ap.add_argument("--image_roots", type=str, nargs="+", default=[],
                    help="folder(s) containing v7w_*.jpg (or a parent with train/val/test sub-folders)")
    ap.add_argument("--out_dir", type=str, default="data/visual7w")
    ap.add_argument("--splits", type=str, nargs="*", default=None, help="restrict to these splits (default: all)")
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--num_workers", type=int, default=8)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--fp16", action="store_true", help="store features as float16")
    ap.add_argument("--normalize", action="store_true", help="L2-normalise features")
    ap.add_argument("--num_shards", type=int, default=1, help="split the image list over several jobs")
    ap.add_argument("--shard_id", type=int, default=0)
    ap.add_argument("--merge", action="store_true", help="merge shard files in --out_dir into clipvitl14.pth")
    return ap.parse_args()


if __name__ == "__main__":
    args = get_args()
    if args.merge:
        merge(args)
    else:
        if not args.image_roots:
            raise SystemExit("--image_roots is required for extraction")
        extract(args)
