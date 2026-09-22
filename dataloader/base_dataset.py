"""
dataloader/base_dataset.py
==========================
Shared tokenisation logic for the multiple-choice VideoQA datasets.

BaseDataset
    _get_text_token(text, answer)
        Encodes one sample with the three Flipped-VQA templates (vqa / vaq /
        qav), pads every sequence to `max_seq_len` and builds:
          text_id     token ids (video placeholders -> 0)
          label       supervision targets (0 = ignore; qav uses -1 = ignore
                      and frame indices 0..max_feats-1 as targets)
          video_start position of the first video placeholder
          video_index positions of the video placeholders (used by qav)
    _sample_video(video)
        Uniformly sub-samples / zero-pads a (T, 768) feature tensor to
        exactly `max_feats` frames.

Subclasses must set `self.answer_mapping` (e.g. {0: '(A)', ...}) and
implement `_get_text(idx)` returning a dict with keys
q_text / o_text / a_text / options.
"""

import copy

import torch
from torch.utils.data import Dataset


class BaseDataset(Dataset):
    def __init__(self, args, tokenizer, split):
        self.args = args
        self.max_feats = args.max_feats
        self.features_dim = 768
        self.tokenizer = tokenizer
        self.max_seq_len = args.max_seq_len
        self.split = split

    # ── video ──

    def _sample_video(self, video: torch.Tensor):
        """(T, 768) -> (max_feats, 768) by uniform sub-sampling or zero padding."""
        if len(video) > self.max_feats:
            idx = [(j * len(video)) // self.max_feats for j in range(self.max_feats)]
            video = video[idx]
            video_len = self.max_feats
        elif len(video) < self.max_feats:
            video_len = len(video)
            video = torch.cat([video, torch.zeros(self.max_feats - video_len, self.features_dim)], dim=0)
        else:
            video_len = self.max_feats
        return video, video_len

    # ── text ──

    def _get_padding_id(self, text_id):
        padded = torch.full((len(text_id), self.max_seq_len), -1, dtype=torch.int64)
        for i, tid in enumerate(text_id):
            if len(tid) > self.max_seq_len:
                tid = tid[: self.max_seq_len]
                print('[BaseDataset] max sequence length overflow, sequence truncated')
            padded[i, : len(tid)] = tid.clone().detach().to(torch.int64)
        return padded

    def _get_text_token(self, text, answer):
        tok = self.tokenizer
        vqa_id, vqa_prefix_index, vqa_video_start = tok.encode_vqa(
            text=text, max_feats=self.max_feats, split=self.split, answer_mapping=self.answer_mapping, answer=answer)
        vaq_id, vaq_prefix_index, vaq_video_start = tok.encode_vaq(
            text=text, max_feats=self.max_feats, split=self.split, answer_mapping=self.answer_mapping, answer=answer)
        qav_id, qav_prefix_index = tok.encode_qav(
            text=text, max_feats=self.max_feats, split=self.split, answer_mapping=self.answer_mapping, answer=answer)

        vqa_id = self._get_padding_id([torch.tensor(v, dtype=torch.int64) for v in vqa_id])
        vaq_id = self._get_padding_id([torch.tensor(v, dtype=torch.int64) for v in vaq_id])
        qav_id = self._get_padding_id([torch.tensor(v, dtype=torch.int64) for v in qav_id])

        # labels: everything before the prefix is ignored (set to 0)
        vqa_label = copy.deepcopy(vqa_id)
        vqa_label[:, :vqa_prefix_index] = -1
        vqa_label[~vqa_label.ge(0)] = 0

        vaq_label = copy.deepcopy(vaq_id)
        vaq_label[:, :vaq_prefix_index] = -1
        vaq_label[~vaq_label.ge(0)] = 0

        # qav: predict the frame index at each video placeholder position
        qav_label = torch.ones_like(qav_id) * -1
        effective_feats = min(self.max_feats, qav_id.size(1) - qav_prefix_index)
        qav_label[:, qav_prefix_index:qav_prefix_index + effective_feats] = torch.arange(effective_feats)

        # video placeholders (-2) and padding (-1) -> token id 0
        vqa_id[~vqa_id.ge(0)] = 0
        vaq_id[~vaq_id.ge(0)] = 0
        qav_id[~qav_id.ge(0)] = 0

        # positions of the video placeholders
        vqa_video_index = torch.arange(vqa_prefix_index, vqa_prefix_index + self.max_feats)
        vaq_video_index = torch.arange(vaq_prefix_index, vaq_prefix_index + self.max_feats)
        effective_video_feats = min(self.max_feats, max(0, self.max_seq_len - qav_prefix_index))
        if effective_video_feats > 0:
            qav_video_index = torch.arange(qav_prefix_index, qav_prefix_index + effective_video_feats)
            if effective_video_feats < self.max_feats:
                pad = torch.full((self.max_feats - effective_video_feats,), self.max_seq_len - 1,
                                 dtype=qav_video_index.dtype)
                qav_video_index = torch.cat([qav_video_index, pad])
        else:
            qav_video_index = torch.full((self.max_feats,), self.max_seq_len - 1, dtype=torch.int64)

        text_id = {'vqa': vqa_id, 'vaq': vaq_id, 'qav': qav_id}
        label = {'vqa': vqa_label, 'vaq': vaq_label, 'qav': qav_label}
        video_start = {'vqa': vqa_video_start, 'vaq': vaq_video_start, 'qav': qav_prefix_index}
        video_index = {'vqa': vqa_video_index, 'vaq': vaq_video_index, 'qav': qav_video_index}
        return text_id, label, video_start, video_index
