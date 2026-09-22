"""
llama/tokenizer.py
==================
SentencePiece tokenizer for LLaMA plus the multiple-choice VideoQA prompt
templates of Flipped-VQA.

Every encoder returns token-id lists in which the video frames are marked
with the placeholder id -2 (replaced by projected video features inside the
model), together with the index of the first token that carries a
supervision label (`prefix_index`) and, where relevant, the position of the
first video placeholder (`video_start`).

Methods
-------
encode(s, bos, eos)            plain text encoding
encode_vqa(...)                 "Video + Question -> Answer" (main task)
encode_vaq(...)                 "Video + Answer -> Question" (auxiliary, --vaq)
encode_qav(...)                 "Question + Answer -> Video" (auxiliary, --qav)
decode(ids)                     ids -> text

At training time only the correct option is encoded; at evaluation time one
sequence per option is produced so the model can score every candidate.
"""

import os
from logging import getLogger
from typing import List

from sentencepiece import SentencePieceProcessor

logger = getLogger()

VIDEO_PLACEHOLDER = -2


class Tokenizer:
    def __init__(self, model_path: str):
        assert os.path.isfile(model_path), model_path
        self.sp_model = SentencePieceProcessor(model_file=model_path)
        logger.info(f"Reloaded SentencePiece model from {model_path}")

        self.n_words: int = self.sp_model.vocab_size()
        self.bos_id: int = self.sp_model.bos_id()
        self.eos_id: int = self.sp_model.eos_id()
        self.pad_id: int = self.sp_model.pad_id()

        # ids of the words "Video", "Question", "Answer" and "\n" in the LLaMA vocabulary,
        # used to locate the start of the supervised span in each template
        self.v_token_id = 15167
        self.q_token_id = 16492
        self.a_token_id = 22550
        self.nl_id = 13
        logger.info(f"#words: {self.n_words} - BOS ID: {self.bos_id} - EOS ID: {self.eos_id}")
        assert self.sp_model.vocab_size() == self.sp_model.get_piece_size()

    def encode(self, s: str, bos: bool, eos: bool) -> List[int]:
        assert isinstance(s, str)
        t = self.sp_model.encode(s)
        if bos:
            t = [self.bos_id] + t
        if eos:
            t = t + [self.eos_id]
        return t

    def decode(self, t: List[int]) -> str:
        return self.sp_model.decode(t)

    # ── prompt templates ──

    def encode_vqa(self, text=None, max_feats=10, split='train', answer_mapping=None, answer=None):
        """Video + question + choices -> answer letter. Returns (ids, prefix_index, video_start)."""
        i_text = "Instruction: Predict the answer based on the video and question.\n"
        s1 = i_text + 'Video:'
        t1 = [self.bos_id] + self.sp_model.encode(s1)
        video_start = len(t1)
        video = [VIDEO_PLACEHOLDER] * max_feats

        s2 = text['q_text'] + text['o_text'] + text['a_text']
        if split == 'train':
            t2 = self.sp_model.encode(s2 + answer_mapping[answer]) + [self.eos_id]
            t = [t1 + video + [self.nl_id] + t2]
            prefix_index = t[0].index(self.a_token_id) + 5
        else:
            t = []
            for _, v in answer_mapping.items():
                t2 = self.sp_model.encode(s2 + v) + [self.eos_id]
                t.append(t1 + video + [self.nl_id] + t2)
            prefix_index = t[answer].index(self.a_token_id) + 5
        return t, prefix_index, video_start

    def encode_vaq(self, text=None, max_feats=10, split='train', answer_mapping=None, answer=None):
        """Video + choices + answer -> question. Returns (ids, prefix_index, video_start)."""
        i_text = "Instruction: Predict the question based on the video and answer.\n"
        q_text = text['q_text'].strip()
        s1 = i_text + 'Video:'
        t1 = [self.bos_id] + self.sp_model.encode(s1)
        video_start = len(t1)
        video = [VIDEO_PLACEHOLDER] * max_feats

        s2 = text['o_text'] + text['a_text']
        if split == 'train':
            t2 = self.sp_model.encode(s2 + answer_mapping[answer] + "\n" + q_text) + [self.eos_id]
            t = [t1 + video + [self.nl_id] + t2]
            prefix_index = t[0].index(self.q_token_id) + 2
        else:
            t = []
            for _, v in answer_mapping.items():
                t2 = self.sp_model.encode(s2 + v + "\n" + q_text) + [self.eos_id]
                t.append(t1 + video + [self.nl_id] + t2)
            prefix_index = t[answer].index(self.q_token_id) + 2
        return t, prefix_index, video_start

    def encode_qav(self, text=None, max_feats=10, split='train', answer_mapping=None, answer=None):
        """Question + choices + answer -> video. Returns (ids, prefix_index)."""
        i_text = "Instruction: Predict the video based on the question and answer.\n"
        s1 = i_text + text['q_text'] + text['o_text'] + text['a_text']
        video = [VIDEO_PLACEHOLDER] * max_feats

        if split == 'train':
            t1 = [self.bos_id] + self.sp_model.encode(s1 + answer_mapping[answer] + "\n" + "Video:")
            t = [t1 + video + [self.eos_id]]
            prefix_index = t[0].index(self.v_token_id) + 2
        else:
            t = []
            for _, v in answer_mapping.items():
                t1 = [self.bos_id] + self.sp_model.encode(s1 + v + "\n" + "Video:") + video + [self.eos_id]
                t.append(t1)
            prefix_index = t[answer].index(self.v_token_id) + 2
        return t, prefix_index
