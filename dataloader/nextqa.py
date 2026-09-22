"""
dataloader/nextqa.py
====================
NExT-QA (5-way multiple choice) in the domain-incremental setting: every
question type is one continual-learning task.

Question types / task order used in the paper:
    TP, CW, DC, TC, DL, DO, TN, CH
    (T = temporal, C = causal, D = descriptive)

Files expected under `<data_root>/nextqa/`:
    split_data/{train,val}_{TYPE}.csv    per-type question splits (provided in this repo)
    clipvitl14.pth                       dict video_id -> (T, 768) CLIP ViT-L/14 frame features
                                         (from Flipped-VQA)

Each item returned by __getitem__ (see BaseDataset for the text fields):
    video (max_feats, 768), text_id, label, video_start, video_index,
    answer (int), qtype (int), qtype_str, task_key (== qtype_str), vid, qid
"""

import pandas as pd
import torch

from .base_dataset import BaseDataset

NEXTQA_QTYPES = ['TP', 'CW', 'DC', 'TC', 'DL', 'DO', 'TN', 'CH']


class NextQA(BaseDataset):
    def __init__(self, args=None, tokenizer=None, split='train', type=None, task_key=None):
        super().__init__(args, tokenizer, split)
        self.task_key = task_key or type   # name of this task in the task bank
        root = f'{args.data_root}/nextqa'
        self.data = pd.read_csv(f'{root}/split_data/{split}_{type}.csv')
        self.features = torch.load(f'{root}/clipvitl14.pth', weights_only=True)

        self.answer_mapping = {0: '(A)', 1: '(B)', 2: '(C)', 3: '(D)', 4: '(E)'}
        self.num_options = 5
        self.qtype_mapping = {q: i for i, q in enumerate(NEXTQA_QTYPES)}
        print(f"[NextQA] {split}/{type}: {len(self.data)} samples")

    def _get_text(self, idx):
        question = str(self.data["question"].values[idx]).capitalize().strip()
        if question[-1] != "?":
            question = question + "?"
        options = [self.data[f'a{i}'].values[idx] for i in range(self.num_options)]
        qtype = self.data['type'].values[idx]

        q_text = f"Question Type: {qtype}\n Question: {question}\n"
        o_text = "Choices: \n"
        for i in range(self.num_options):
            o_text += f"{self.answer_mapping[i]} {options[i]}\n"
        a_text = "Answer: The answer is "
        return {'q_text': q_text, 'o_text': o_text, 'a_text': a_text, 'options': options}

    def _get_video(self, video_id):
        if video_id not in self.features:
            print(f"[NextQA] video {video_id} not in features, using zeros")
            video = torch.zeros(1, self.features_dim)
        else:
            video = self.features[video_id].float()
        return self._sample_video(video)

    def __getitem__(self, idx):
        vid = self.data['video'].values[idx]
        qtype_str = self.data['type'].values[idx]
        answer = int(self.data['answer'].values[idx])

        text = self._get_text(idx)
        text_id, label, video_start, video_index = self._get_text_token(text, answer)
        video, video_len = self._get_video(f'{vid}')

        return {
            "vid": vid,
            "qid": idx,
            "video": video,
            "video_len": video_len,
            "text_id": text_id,
            "label": label,
            "video_start": video_start,
            "video_index": video_index,
            "answer": answer,
            "qtype": self.qtype_mapping[qtype_str],
            "qtype_str": qtype_str,
            "task_key": self.task_key,
        }

    def __len__(self):
        return len(self.data)
