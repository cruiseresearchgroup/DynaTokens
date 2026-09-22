"""
util/lr_sched.py
================
Learning-rate schedule: linear warm-up followed by half-cycle cosine decay
(from MAE). Every optimizer parameter group is scheduled relative to its own
`initial_lr`, so per-component learning rates (token generator, z_inv, z_sp,
gates) keep their ratios throughout training.
"""

import math


def adjust_learning_rate(optimizer, epoch, args):
    """Set the LR of every param group for fractional `epoch`; returns nothing."""
    for param_group in optimizer.param_groups:
        base_lr = param_group['initial_lr']
        if epoch < args.warmup_epochs:
            lr = base_lr * epoch / args.warmup_epochs
        else:
            progress = (epoch - args.warmup_epochs) / max(1e-8, (args.epochs - args.warmup_epochs))
            lr = args.min_lr + (base_lr - args.min_lr) * 0.5 * (1. + math.cos(math.pi * progress))
        param_group["lr"] = lr * param_group.get("lr_scale", 1.0)
