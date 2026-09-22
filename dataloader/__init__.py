"""
dataloader package
==================
Datasets and DataLoader construction for continual VideoQA.

    base_dataset.py   shared tokenisation / video sampling (BaseDataset)
    nextqa.py         NExT-QA, one task per question type
    dramaqa.py        DramaQA, one task per question type
    visual7w.py       Visual7W-telling, a single pre-training task ("visualqa")

Public API
----------
DATASETS              dataset name -> (dataset class, default task order)
BENCHMARKS            benchmark name (the --dataset argument) -> list of task
                      specs (dataset, question type, task key); a benchmark can
                      chain several datasets (e.g. 'nextqa_dramaqa')
build_task_list(benchmark, task_order=None)
                      task specs for a benchmark, optionally re-ordered
load_data(args, tokenizer, split, task, distributed=True)
                      DataLoader for one task spec
batch_collate         collate function producing the batch dict consumed by the model
"""

from collections import namedtuple

import torch

from util import misc
from .dramaqa import DramaQA, DRAMAQA_QTYPES
from .nextqa import NextQA, NEXTQA_QTYPES
from .visual7w import Visual7W, VISUAL7W_TASK

DATASETS = {
    'nextqa': (NextQA, NEXTQA_QTYPES),
    'dramaqa': (DramaQA, DRAMAQA_QTYPES),
    'visual7w': (Visual7W, [VISUAL7W_TASK]),
}

# A task: which dataset, which question type, and the key under which it is
# stored in the task bank. Single-dataset benchmarks use the bare question
# type as key; multi-dataset benchmarks use "<dataset>:<type>".
TaskSpec = namedtuple('TaskSpec', ['dataset', 'qtype', 'key'])

BENCHMARKS = {
    'nextqa': [TaskSpec('nextqa', q, q) for q in NEXTQA_QTYPES],
    'dramaqa': [TaskSpec('dramaqa', q, q) for q in DRAMAQA_QTYPES],
    'visual7w': [TaskSpec('visual7w', VISUAL7W_TASK, VISUAL7W_TASK)],
    'nextqa_dramaqa': ([TaskSpec('nextqa', q, f'nextqa:{q}') for q in NEXTQA_QTYPES]
                       + [TaskSpec('dramaqa', q, f'dramaqa:{q}') for q in DRAMAQA_QTYPES]),
}


def build_task_list(benchmark: str, task_order=None):
    """
    Task specs of `benchmark`. `task_order` (comma-separated task keys, e.g.
    "TP,CW" or "nextqa:TP,dramaqa:TW") selects / re-orders the tasks.
    """
    tasks = list(BENCHMARKS[benchmark])
    if task_order:
        by_key = {t.key: t for t in tasks}
        wanted = [k.strip() for k in task_order.split(',') if k.strip()]
        unknown = [k for k in wanted if k not in by_key]
        if unknown:
            raise ValueError(f"Unknown task(s) {unknown} for benchmark '{benchmark}'; known: {list(by_key)}")
        tasks = [by_key[k] for k in wanted]
    return tasks


def load_data(args, tokenizer, split, task: TaskSpec, distributed=True):
    """
    DataLoader for one task.

    distributed=True  : DistributedSampler (each rank sees its own shard)
    distributed=False : plain sequential loader over the full split
                        (used for validation on rank 0)
    """
    is_train = (split == 'train')
    ds_cls, _ = DATASETS[task.dataset]
    dataset = ds_cls(args=args, tokenizer=tokenizer, split=split, type=task.qtype, task_key=task.key)

    if distributed:
        sampler = torch.utils.data.DistributedSampler(
            dataset, num_replicas=misc.get_world_size(), rank=misc.get_rank(),
            shuffle=is_train, drop_last=is_train,
        )
    else:
        sampler = None

    return torch.utils.data.DataLoader(
        dataset,
        sampler=sampler,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        collate_fn=batch_collate,
        pin_memory=args.pin_mem,
        shuffle=(sampler is None and is_train),
        persistent_workers=(args.num_workers > 0),
        drop_last=is_train,
    )


def batch_collate(batch):
    bs = len(batch)

    def stack(field, sub):
        return torch.stack([batch[i][field][sub] for i in range(bs)])

    return {
        "vid": [b["vid"] for b in batch],
        "qid": [b["qid"] for b in batch],
        "video": torch.stack([b["video"] for b in batch]),
        "video_len": torch.tensor([b["video_len"] for b in batch], dtype=torch.long),
        "text_id": {k: stack('text_id', k) for k in ('vqa', 'vaq', 'qav')},
        "label": {k: stack('label', k) for k in ('vqa', 'vaq', 'qav')},
        "video_index": {k: stack('video_index', k) for k in ('vqa', 'vaq', 'qav')},
        "video_start": {k: [b["video_start"][k] for b in batch] for k in ('vqa', 'vaq', 'qav')},
        "answer": torch.tensor([b["answer"] for b in batch], dtype=torch.long),
        "qtype": torch.tensor([b["qtype"] for b in batch], dtype=torch.long),
        "qtype_str": [b["qtype_str"] for b in batch],
        "task_key": [b["task_key"] for b in batch],
    }
