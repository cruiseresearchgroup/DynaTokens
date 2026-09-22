"""
dataloader/visual7w.py
======================
Visual7W-telling (image QA, 4-way multiple choice) used as a single
*pre-training task* named `visualqa` (see scripts/run_visual7w.sh and
scripts/run_nextqa_from_visual7w.sh).

The image is treated as a "video" by repeating its CLIP feature `max_feats`
times, so the sample format is identical to NExT-QA and the same model /
prompt templates can be used. Options are 3 distractors + the correct
answer, shuffled deterministically per `qa_id`.

Files expected under `<data_root>/visual7w/`:
    train.json, val.json    80/20 image-level split of Visual7W-telling
                            (`{"images": [{"image_id", "qa_pairs": [...]}, ...]}`,
                            provided in this repo)
    clipvitl14.pth          dict image_id -> (768,) CLIP ViT-L/14 image feature
                            (optionally wrapped as {"features": {...}})

Because `max_seq_len` is kept at the NExT-QA value (128), questions / options
that would not fit are truncated word- then character-wise until the three
prompt templates fit (see _make_text_safe).
"""

import json

import numpy as np
import torch

from .base_dataset import BaseDataset

VISUAL7W_TASK = 'visualqa'
VISUAL7W_QTYPES = {"what": 0, "where": 1, "when": 2, "who": 3, "why": 4, "how": 5}


def _deterministic_shuffle(options, qa_id: int):
    """Shuffle `options` with a permutation that depends only on qa_id."""
    rng = np.random.RandomState((int(qa_id) * 1000003 + 17) % (2 ** 32 - 1))
    perm = np.arange(len(options))
    rng.shuffle(perm)
    return [options[i] for i in perm], perm


class Visual7W(BaseDataset):
    def __init__(self, args=None, tokenizer=None, split="train", type=None, task_key=None):
        super().__init__(args, tokenizer, split)
        if split not in ("train", "val"):
            raise ValueError("Visual7W split must be 'train' or 'val'")
        root = f'{args.data_root}/visual7w'
        self.task_key = task_key or VISUAL7W_TASK

        with open(f'{root}/{split}.json', "r") as f:
            obj = json.load(f)
        self.data = []
        for im in obj.get("images", []):
            for qa in im.get("qa_pairs", []):
                self.data.append({
                    "image_id": int(qa.get("image_id", im["image_id"])),
                    "qa_id": int(qa["qa_id"]),
                    "question": qa["question"],
                    "answer": qa["answer"],
                    "multiple_choices": qa.get("multiple_choices", []),
                    "type": str(qa.get("type", "what")).lower(),
                })

        feats = torch.load(f'{root}/clipvitl14.pth', map_location="cpu", weights_only=False)
        self.features = feats["features"] if isinstance(feats, dict) and "features" in feats else feats

        self.num_options = 4
        self.answer_mapping = {0: "(A)", 1: "(B)", 2: "(C)", 3: "(D)"}
        self.qtype_mapping = VISUAL7W_QTYPES
        print(f"[Visual7W] {split}: {len(self.data)} QA pairs, {len(self.features)} image features")

    # ── video (repeated image feature) ──

    def _get_video(self, image_id: int):
        feat = self.features.get(image_id, self.features.get(str(image_id)))
        if feat is None:
            feat = torch.zeros(self.features_dim)
        feat = feat.float() if torch.is_tensor(feat) else torch.tensor(feat, dtype=torch.float32)
        return feat.view(1, -1).repeat(self.max_feats, 1).contiguous(), self.max_feats

    # ── options ──

    def _build_options(self, row):
        correct = str(row["answer"]).strip()
        distractors = [str(x).strip() for x in row["multiple_choices"] if str(x).strip() and str(x).strip() != correct]
        distractors = (distractors + ["None"] * 3)[:3]
        options, perm = _deterministic_shuffle(distractors + [correct], row["qa_id"])
        return options, int(np.where(perm == 3)[0][0])

    # ── text (with truncation so that every template fits max_seq_len) ──

    def _format_text(self, question, qtype, options):
        question = str(question).strip().capitalize()
        if not question.endswith("?"):
            question += "?"
        q_text = f"Question Type: {qtype}\n Question: {question}\n"
        o_text = "Choices: \n" + "".join(f"{self.answer_mapping[i]} {options[i]}\n" for i in range(self.num_options))
        return {"q_text": q_text, "o_text": o_text, "a_text": "Answer: The answer is ", "options": options}

    def _fits(self, text, answer) -> bool:
        enc = dict(text=text, max_feats=self.max_feats, split=self.split, answer_mapping=self.answer_mapping, answer=answer)
        vqa, vqa_p, _ = self.tokenizer.encode_vqa(**enc)
        vaq, vaq_p, _ = self.tokenizer.encode_vaq(**enc)
        qav, qav_p = self.tokenizer.encode_qav(**enc)
        max_prefix = max(1, self.max_seq_len - self.max_feats)
        return (max(len(x) for x in vqa + vaq + qav) <= self.max_seq_len
                and max(vqa_p, vaq_p, qav_p) <= max_prefix)

    @staticmethod
    def _truncate_words(s, n):
        ws = str(s).strip().split()
        return str(s).strip() if len(ws) <= n else " ".join(ws[:n]) + " ..."

    @staticmethod
    def _truncate_chars(s, n):
        s = str(s).strip()
        if len(s) <= n:
            return s
        cut = s[:n]
        return (cut.rsplit(" ", 1)[0] if " " in cut else cut) + " ..."

    def _make_text_safe(self, question, qtype, options, answer):
        text = self._format_text(question, qtype, options)
        if self._fits(text, answer):
            return text
        for trunc, q_caps, o_caps in (
            (self._truncate_words, [40, 32, 28, 24, 20, 16, 12, 10, 8, 6, 4, 3], [18, 14, 12, 10, 8, 6, 5, 4, 3, 2]),
            (self._truncate_chars, [200, 160, 120, 96, 80, 64, 48, 40, 32, 24], [96, 80, 64, 48, 40, 32, 24, 20, 16]),
        ):
            for qc in q_caps:
                for oc in o_caps:
                    text = self._format_text(trunc(question, qc), qtype, [trunc(o, oc) for o in options])
                    if self._fits(text, answer):
                        return text
        return self._format_text("Choose the correct answer", qtype, ["A", "B", "C", "D"])

    # ── dataset ──

    def __getitem__(self, idx):
        row = self.data[idx]
        options, answer = self._build_options(row)
        text = self._make_text_safe(row["question"], row["type"], options, answer)
        text_id, label, video_start, video_index = self._get_text_token(text, answer)
        video, video_len = self._get_video(row["image_id"])

        return {
            "vid": row["image_id"],
            "qid": row["qa_id"],
            "video": video,
            "video_len": video_len,
            "text_id": text_id,
            "label": label,
            "video_start": video_start,
            "video_index": video_index,
            "answer": answer,
            "qtype": self.qtype_mapping.get(row["type"], 0),   # what/where/... (for reference only)
            "qtype_str": VISUAL7W_TASK,
            "task_key": self.task_key,
        }

    def __len__(self):
        return len(self.data)
