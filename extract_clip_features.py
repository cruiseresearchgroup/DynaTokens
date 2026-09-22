#!/usr/bin/env python3
"""
Extract CLIP ViT-L/14 image features for Visual7W-Telling (your JSON format) and save.

Outputs:
- per-shard tensor dump: visual7w_telling_clip_vitl14_shardXXX_of_YYY.pt
- per-shard metadata:    visual7w_telling_clip_vitl14_shardXXX_of_YYY.json

Optional:
- --save_dict: additionally save a dictionary-format .pth:
    clip-l-14-visual-7w.pth (or your chosen name)
  containing:
    {
      "features": { image_id: Tensor(D,) },
      "filenames": { image_id: str },
      "meta": {...}
    }
  Note: dict has Python overhead; tensor format is more IO-efficient.
"""

import json
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from tqdm import tqdm

import open_clip


# ---------------- JSON + path utils ----------------

def load_image_records(dataset_json: str) -> List[Dict]:
    """
    Expected format:
    {
      "images": [
        {
          "image_id": ...,
          "split": "train"/"val"/"test",
          "filename": "v7w_....jpg",
          "qa_pairs": [...]
        }, ...
      ]
    }
    """
    with open(dataset_json, "r") as f:
        data = json.load(f)
    if "images" not in data or not isinstance(data["images"], list):
        raise ValueError("Expected top-level key 'images' as a list in dataset JSON.")
    return data["images"]


def resolve_v7w_image_path(img_rec: Dict, image_roots: List[str]) -> str:
    """
    Try common layouts:
      1) <root>/<filename>
      2) <root>/<split>/<filename>
    """
    fname = img_rec.get("filename")
    if not fname:
        raise ValueError(f"Missing 'filename' in record keys={list(img_rec.keys())}")

    split = img_rec.get("split", None)  # train/val/test or None

    candidates: List[Path] = []
    for root in image_roots:
        rootp = Path(root)
        candidates.append(rootp / fname)
        if split:
            candidates.append(rootp / split / fname)

    for p in candidates:
        if p.exists():
            return str(p)

    tried = "\n".join([str(c) for c in candidates[:10]])
    raise FileNotFoundError(
        f"Cannot find image file '{fname}' (split={split}).\n"
        f"Tried (first 10):\n{tried}\n"
        f"Hint: set --image_roots to the folder that directly contains these files, "
        f"or a parent folder with train/val/test subfolders."
    )


# ---------------- Dataset ----------------

class ImagePathDataset(Dataset):
    def __init__(self, items: List[Tuple[int, str, str]], preprocess):
        """
        items: (image_id, filepath, filename)
        """
        self.items = items
        self.preprocess = preprocess

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx: int):
        image_id, path, filename = self.items[idx]
        img = Image.open(path).convert("RGB")
        x = self.preprocess(img)
        return image_id, filename, x


# ---------------- Main ----------------

@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_json", type=str, required=True)
    ap.add_argument(
        "--image_roots",
        type=str,
        nargs="+",
        required=True,
        help="directories containing v7w_*.jpg, or parent with train/val/test subfolders",
    )
    ap.add_argument("--out_dir", type=str, required=True)

    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--num_workers", type=int, default=8)
    ap.add_argument("--device", type=str, default="cuda")

    ap.add_argument("--fp16", action="store_true", help="store features as float16 (recommended)")
    ap.add_argument("--normalize", action="store_true", help="L2-normalize features (recommended)")

    # optional: filter splits
    ap.add_argument(
        "--splits",
        type=str,
        nargs="*",
        default=None,
        help="e.g. --splits train val test ; default=None means all",
    )

    # shard options (multi-GPU / many jobs)
    ap.add_argument("--num_shards", type=int, default=1)
    ap.add_argument("--shard_id", type=int, default=0)

    # optional: also save dict-format .pth
    ap.add_argument(
        "--save_dict",
        action="store_true",
        help="also save {image_id->feature} dict as a .pth (per shard)",
    )
    ap.add_argument(
        "--dict_name",
        type=str,
        default=None,
        help="filename for dict .pth (default: clip-l-14-visual-7w_<shard>.pth)",
    )

    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) load records
    records = load_image_records(args.dataset_json)

    # 2) optional split filter
    if args.splits is not None and len(args.splits) > 0:
        keep = set(args.splits)
        records = [r for r in records if r.get("split") in keep]
        print(f"[INFO] Filtered by splits={args.splits}: {len(records)} images remain.")

    # 3) resolve paths (unique by image_id)
    items: List[Tuple[int, str, str]] = []
    seen = set()

    for rec in tqdm(records, desc="Resolving image paths"):
        image_id = rec.get("image_id")
        if image_id is None:
            raise ValueError(f"Missing 'image_id' in record: keys={list(rec.keys())}")
        image_id = int(image_id)

        if image_id in seen:
            continue
        seen.add(image_id)

        path = resolve_v7w_image_path(rec, args.image_roots)
        items.append((image_id, path, rec.get("filename", "")))

    items.sort(key=lambda x: x[0])
    print(f"[INFO] Resolved {len(items)} unique images.")

    # 4) shard selection
    if not (0 <= args.shard_id < args.num_shards):
        raise ValueError("shard_id must be in [0, num_shards).")
    shard_items = items[args.shard_id :: args.num_shards]
    print(f"[INFO] Shard {args.shard_id}/{args.num_shards}: {len(shard_items)} images.")

    # 5) load CLIP ViT-L/14
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model, _, preprocess = open_clip.create_model_and_transforms(
        model_name="ViT-L-14",
        pretrained="openai",
    )
    model = model.to(device).eval()

    # 6) dataloader
    ds = ImagePathDataset(shard_items, preprocess)
    dl = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    if len(ds) == 0:
        raise RuntimeError("No images found after resolving paths. Check --image_roots and JSON.")

    # 7) infer feature dim
    image_ids0, fnames0, x0 = next(iter(dl))
    x0 = x0.to(device, non_blocking=True)
    feats0 = model.encode_image(x0)
    feat_dim = int(feats0.shape[-1])
    print(f"[INFO] Feature dim = {feat_dim}")  # typically 768 for ViT-L/14

    # 8) allocate output tensors (tensor format)
    dtype = torch.float16 if args.fp16 else torch.float32
    feats_cpu = torch.empty((len(ds), feat_dim), dtype=dtype, device="cpu")
    ids_cpu = torch.empty((len(ds),), dtype=torch.int64, device="cpu")
    fnames_list: List[str] = [""] * len(ds)

    # (optional) dict format
    feat_dict: Optional[Dict[int, torch.Tensor]] = {} if args.save_dict else None
    fname_dict: Optional[Dict[int, str]] = {} if args.save_dict else None

    # 9) extraction loop
    offset = 0
    for image_ids, fnames, x in tqdm(dl, desc="Extracting CLIP features"):
        b = x.shape[0]
        x = x.to(device, non_blocking=True)

        feats = model.encode_image(x)  # (B, D), float32 on GPU
        if args.normalize:
            feats = feats / feats.norm(dim=-1, keepdim=True).clamp(min=1e-12)

        feats = feats.detach().to("cpu")
        if dtype == torch.float16:
            feats = feats.half()

        feats_cpu[offset:offset + b].copy_(feats)
        ids_cpu[offset:offset + b].copy_(image_ids.to("cpu"))

        # filenames are Python strings (collated by DataLoader)
        for i in range(b):
            fnames_list[offset + i] = fnames[i]

        # also populate dict if requested
        if args.save_dict:
            for i in range(b):
                iid = int(image_ids[i].item())
                # store a 1D tensor (D,)
                feat_dict[iid] = feats[i].contiguous()
                fname_dict[iid] = fnames[i]

        offset += b

    assert offset == len(ds)

    # 10) save
    shard_tag = f"shard{args.shard_id:03d}_of_{args.num_shards:03d}"
    feat_path = out_dir / f"visual7w_telling_clip_vitl14_{shard_tag}.pt"
    meta_path = out_dir / f"visual7w_telling_clip_vitl14_{shard_tag}.json"

    id2idx = {int(ids_cpu[i].item()): int(i) for i in range(len(ds))}
    fname2idx = {fnames_list[i]: int(i) for i in range(len(ds))}

    meta = {
        "model": "CLIP ViT-L/14 (openai)",
        "feature_dim": int(feat_dim),
        "num_images": int(len(ds)),
        "fp16": bool(args.fp16),
        "normalized": bool(args.normalize),
        "shard_id": int(args.shard_id),
        "num_shards": int(args.num_shards),
    }

    # tensor format
    torch.save(
        {
            "image_ids": ids_cpu,      # (N,)
            "filenames": fnames_list,  # list[str]
            "features": feats_cpu,     # (N, D)
            "meta": meta,
        },
        feat_path,
    )
    with open(meta_path, "w") as f:
        json.dump({"meta": meta, "id2idx": id2idx, "fname2idx": fname2idx}, f)

    print(f"[DONE] Saved tensor features: {feat_path}")
    print(f"[DONE] Saved metadata:        {meta_path}")

    # dict format (optional)
    if args.save_dict:
        if args.dict_name is None:
            dict_name = f"clip-l-14-visual-7w_{shard_tag}.pth"
        else:
            # if user passes a fixed name, shard it to avoid overwriting
            p = Path(args.dict_name)
            dict_name = f"{p.stem}_{shard_tag}{p.suffix or '.pth'}"

        dict_path = out_dir / dict_name
        torch.save(
            {
                "features": feat_dict,   # dict[int] -> tensor(D,)
                "filenames": fname_dict, # dict[int] -> str
                "meta": meta,
            },
            dict_path,
        )
        print(f"[DONE] Saved dict features:  {dict_path}")


if __name__ == "__main__":
    main()
