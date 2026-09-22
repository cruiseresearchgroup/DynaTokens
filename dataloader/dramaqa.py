"""
dataloader/dramaqa.py
=====================
DramaQA (AnotherMissOh, 5-way multiple choice) in the domain-incremental
setting: every question type is one continual-learning task.

Question types / task order used in the paper:
    TW, DO, DL, CH, CW

Files expected under `<data_root>/dramaqa/`:
    split_data/AnotherMissOhQA_{train,val}_set_{TYPE}.json   (provided in this repo)
    clipvitl14.pth      dict clip_id -> (T, 768) CLIP ViT-L/14 frame features
                        (from Flipped-VQA)

A DramaQA question refers either to a single shot (clip id ending in a
non-zero shot number) or to a whole scene (clip id ending in `0000`); for a
scene the features of all shots in `shot_contained` are concatenated before
sub-sampling to `max_feats` frames.
"""

import json

import pandas as pd
import torch

from .base_dataset import BaseDataset

DRAMAQA_QTYPES = ['TW', 'DO', 'DL', 'CH', 'CW']


class DramaQA(BaseDataset):
    def __init__(self, args=None, tokenizer=None, split='train', type=None, task_key=None):
        super().__init__(args, tokenizer, split)
        self.task_key = task_key or type   # name of this task in the task bank
        root = f'{args.data_root}/dramaqa'
        with open(f'{root}/split_data/AnotherMissOhQA_{split}_set_{type}.json', "r") as f:
            self.data = pd.DataFrame(json.load(f))
        self.data.rename(columns={'vid': 'video', 'correct_idx': 'answer'}, inplace=True)
        self.data['type'] = type
        self.features = torch.load(f'{root}/clipvitl14.pth', weights_only=True)

        self.answer_mapping = {0: '(A)', 1: '(B)', 2: '(C)', 3: '(D)', 4: '(E)'}
        self.num_options = 5
        self.qtype_mapping = {q: i for i, q in enumerate(DRAMAQA_QTYPES)}
        print(f"[DramaQA] {split}/{type}: {len(self.data)} samples")

    def _get_text(self, idx):
        sample = self.data.iloc[idx]
        question = str(sample["que"]).capitalize().strip()
        if question[-1] != "?":
            question = question + "?"
        options = sample['answers']

        q_text = f"Question: {question}\n"
        o_text = "Choices: \n"
        for i in range(self.num_options):
            o_text += f"{self.answer_mapping[i]} {options[i]}\n"
        a_text = "Answer: The answer is "
        return {'q_text': q_text, 'o_text': o_text, 'a_text': a_text, 'options': options}

    def _get_video(self, video_id, idx):
        if video_id[-4:] == '0000':
            # scene: concatenate the features of every contained shot
            start, end = self.data.iloc[idx]['shot_contained']
            chunks = []
            for i in range(start, end + 1):
                v_name = video_id[:-4] + f'{i:04}'
                if v_name not in self.features:
                    print(f"[DramaQA] shot {v_name} not in features, using zeros")
                    chunks.append(torch.zeros(1, self.features_dim))
                else:
                    chunks.append(self.features[v_name].float())
            video = torch.cat(chunks, dim=0)
        else:
            if video_id not in self.features:
                print(f"[DramaQA] shot {video_id} not in features, using zeros")
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
        video, video_len = self._get_video(f'{vid}', idx)

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
