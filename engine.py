"""
engine.py
=========
One-epoch training and validation loops.

train_one_epoch(...)
    Optimises the total objective of Eq. 16:

        L = L_NLL + lambda_LA * L_LA + lambda_inv * ||z_inv - z_inv*||^2

    L_NLL      = vqa_loss (+ weight_aux_ques_loss * vaq_loss)
                          (+ weight_aux_vid_loss  * qav_loss)
    L_LA       = LookaheadRegularizer.compute_reg_loss   (tasks t > 1 only)
    z_inv term = model.forward()['z_inv_reg']            (tasks t > 1 only)

    Supports gradient accumulation, AMP, gradient clipping, W&B logging and
    calls `hyper.accumulate_importance` after every optimizer step (SI).
    Returns (epoch_stats, global_step).

    Gradient handling: by default gradients are cleared only at the start of
    each epoch and therefore accumulate across optimizer steps within the
    epoch (this reproduces the paper's runs); --zero_grad_per_step switches
    to the standard behaviour.

val_one_epoch(...)
    Scores every answer option by its mean token NLL and picks the lowest.
    With measure_routing=True the routing accuracy (retrieved task == true
    task) and a routing confusion table are also reported.
    Returns a dict of epoch statistics ('acc', optionally 'route_acc', ...).
"""

import math
import sys
from typing import Iterable

import numpy as np
import torch
from torch.amp import autocast

import util.lr_sched as lr_sched
import util.misc as misc


def wandb_safe_log(run, data: dict, commit: bool = True):
    if run is not None and misc.is_main_process():
        run.log(data, commit=commit)


def _amp_dtype(dev):
    use_cuda = torch.cuda.is_available() and dev.type == "cuda"
    if use_cuda and torch.cuda.get_device_capability(dev.index or 0)[0] >= 8:
        return torch.bfloat16
    return torch.float16


# ═══════════════════════════════════════════════════════════════════════════════
# train_one_epoch
# ═══════════════════════════════════════════════════════════════════════════════

def train_one_epoch(
    model: torch.nn.Module,
    data_loader: Iterable,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    loss_scaler,
    args=None,
    current_qtype=None,
    hyper=None,
    task_index: int = 1,
    *,
    wandb_run=None,
    global_step: int = 0,
    wandb_log_freq: int = 50,
):
    model.train()
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = f'Epoch: [{epoch}]'

    print_freq = max(1, int(len(data_loader) / 4))
    accum_iter = max(1, int(getattr(args, "accum_iter", 1)))
    total_batches = len(data_loader)

    dev = next(model.parameters()).device
    use_cuda = torch.cuda.is_available() and dev.type == "cuda"
    device_type = "cuda" if use_cuda else "cpu"
    amp_dtype = _amp_dtype(dev)
    model_unwrapped = model.module if hasattr(model, "module") else model

    max_grad_norm = float(getattr(args, 'max_grad_norm', 1.0))
    weight_aux_ques_loss = getattr(args, 'weight_aux_ques_loss', 0.5)
    weight_aux_vid_loss = getattr(args, 'weight_aux_vid_loss', 0.1)
    use_lookahead = (
        hyper is not None and getattr(hyper, "enabled", False)
        and task_index > 1 and getattr(args, 'alpha', 0.0) > 0
    )

    optimizer.zero_grad(set_to_none=True)

    for data_iter_step, data in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        for k, v in list(data.items()):
            if torch.is_tensor(v):
                data[k] = v.to(dev, non_blocking=True)

        if data_iter_step % accum_iter == 0:
            lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)

        is_last = (data_iter_step + 1 == total_batches)
        do_step = ((data_iter_step + 1) % accum_iter == 0) or is_last

        # ── forward ──
        with autocast(device_type=device_type, dtype=amp_dtype, enabled=use_cuda):
            losses = model(data, question_type=current_qtype, current_epoch=epoch)
            if not isinstance(losses, dict):
                raise TypeError(f"Model forward during training must return a dict, got {type(losses)}")

            loss = losses['vqa_loss']
            if getattr(args, 'vaq', False):
                loss = loss + weight_aux_ques_loss * losses['vaq_loss']
            if getattr(args, 'qav', False):
                loss = loss + weight_aux_vid_loss * losses['qav_loss']

            if task_index > 1:
                lambda_inv = float(losses.get('lambda_inv', getattr(args, 'lambda_inv', 0.1)))
                loss = loss + lambda_inv * losses['z_inv_reg']
                metric_logger.update(z_inv_reg=float(losses['z_inv_reg'].detach().item()))

            L_reg = None
            if use_lookahead:
                L_reg = hyper.compute_reg_loss(model_unwrapped, data, t=task_index)
                loss = loss + float(args.alpha) * L_reg
                metric_logger.update(reg_loss=float(L_reg.detach().item()) * float(args.alpha))

        # ── logging ──
        metric_logger.update(vqa_loss=float(losses['vqa_loss'].detach().item()))
        if getattr(args, 'vaq', False):
            metric_logger.update(vaq_loss=float(losses['vaq_loss'].detach().item()))
        if getattr(args, 'qav', False):
            metric_logger.update(qav_loss=float(losses['qav_loss'].detach().item()))

        loss_value = float(loss.detach().item())
        if not math.isfinite(loss_value):
            print(f"Loss is {loss_value}, stopping training")
            sys.exit(1)

        loss = loss / accum_iter

        # ── backward + step ──
        def _before_step():
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)

        def _after_step():
            if hyper is not None and getattr(hyper, "enabled", False):
                hyper.accumulate_importance(model_unwrapped)

        if hasattr(model, "no_sync") and not do_step:
            with model.no_sync():
                loss_scaler(loss, optimizer, parameters=model.parameters(), update_grad=False)
        else:
            loss_scaler(
                loss, optimizer, parameters=model.parameters(), update_grad=do_step,
                before_step=_before_step if do_step else None,
                after_step=_after_step if do_step else None,
            )
        # NOTE: by default gradients are only cleared at the start of the epoch
        # (see the top of this function), i.e. they keep accumulating across
        # optimizer steps within an epoch (clipped to max_grad_norm each step).
        # This is the configuration used for the results in the paper.
        # Pass --zero_grad_per_step for the standard behaviour.
        if do_step and getattr(args, 'zero_grad_per_step', False):
            optimizer.zero_grad(set_to_none=True)

        metric_logger.update(loss=loss_value, lr=float(optimizer.param_groups[-1]["lr"]))

        # ── W&B per-step ──
        if do_step and wandb_run is not None and (global_step % max(1, wandb_log_freq) == 0):
            log_dict = {
                "train/step": global_step,
                "epoch": epoch + (data_iter_step + 1) / max(1, len(data_loader)),
                "train/task_index": task_index,
                "train/qtype": str(current_qtype) if current_qtype else "",
                "train/loss": loss_value,
                "train/vqa_loss": float(losses["vqa_loss"].detach().item()),
                "train/lr": float(optimizer.param_groups[-1]["lr"]),
                "train/z_inv_reg": float(losses["z_inv_reg"].detach().item()),
            }
            if getattr(args, 'vaq', False):
                log_dict["train/vaq_loss"] = float(losses["vaq_loss"].detach().item())
            if getattr(args, 'qav', False):
                log_dict["train/qav_loss"] = float(losses["qav_loss"].detach().item())
            if L_reg is not None:
                log_dict["train/la_reg_unscaled"] = float(L_reg.detach().item())
                log_dict["train/la_reg_scaled"] = float(L_reg.detach().item()) * float(args.alpha)
            wandb_safe_log(wandb_run, log_dict, commit=True)

        if do_step:
            global_step += 1

    # ── epoch summary ──
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}

    if wandb_run is not None:
        epoch_log = {"epoch": epoch, "train_epoch/task_index": task_index}
        epoch_log.update({f"train_epoch/{k}": v for k, v in stats.items()})
        wandb_safe_log(wandb_run, epoch_log, commit=True)

    return stats, global_step


# ═══════════════════════════════════════════════════════════════════════════════
# val_one_epoch
# ═══════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def val_one_epoch(
    model: torch.nn.Module,
    data_loader,
    epoch: int,
    args=None,
    task_type=None,
    *,
    wandb_run=None,
    global_step: int = None,
    measure_routing: bool = False,
    dist_sync: bool = False,
):
    """
    dist_sync=False : this rank evaluates alone (no collectives; safe to call on
                      rank 0 only while other ranks wait at a barrier)
    dist_sync=True  : every rank evaluates its shard and metrics are all-reduced
    """
    do_dist_sync = dist_sync and torch.distributed.is_available() and torch.distributed.is_initialized()

    model.eval()
    metric_logger = misc.MetricLogger(delimiter="  ")
    header = f'Epoch: [{epoch}]'
    print_freq = max(1, int(len(data_loader) / 4))

    dev = next(model.parameters()).device
    use_cuda = torch.cuda.is_available() and dev.type == "cuda"
    device_type = "cuda" if use_cuda else "cpu"
    amp_dtype = _amp_dtype(dev)

    route_correct_sum = 0
    route_total_sum = 0
    route_confusion_agg = {}

    for data in metric_logger.log_every(data_loader, print_freq, header):
        for k, v in list(data.items()):
            if torch.is_tensor(v):
                data[k] = v.to(dev, non_blocking=True)
        answer = data['answer']
        bsz = int(answer.shape[0])

        with autocast(device_type=device_type, dtype=amp_dtype, enabled=use_cuda):
            forward_kwargs = dict(inference=True, current_epoch=epoch, question_type=task_type)
            if measure_routing:
                logits, routing_stats = model(data, return_routing_stats=True, **forward_kwargs)
            else:
                logits = model(data, **forward_kwargs)
                routing_stats = None

        # mean NLL over the answer tokens of each option; lowest NLL wins
        count = torch.clamp((logits != 0).sum(-1), min=1)
        prediction = (logits.sum(-1) / count).argmin(-1)
        acc = float((answer == prediction).sum().item() / max(bsz, 1))
        metric_logger.update(n=int(bsz), acc=acc)

        if measure_routing and routing_stats is not None:
            route_correct_sum += int(routing_stats.get("route_correct", 0))
            route_total_sum += int(routing_stats.get("route_total", 0))
            for pk, cnt in routing_stats.get("route_confusion", {}).items():
                route_confusion_agg[pk] = route_confusion_agg.get(pk, 0) + cnt

    if do_dist_sync:
        torch.distributed.barrier()
        metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}

    if measure_routing:
        if do_dist_sync:
            t = torch.tensor([route_correct_sum, route_total_sum], device=dev, dtype=torch.float64)
            torch.distributed.all_reduce(t, op=torch.distributed.ReduceOp.SUM)
            route_correct_sum, route_total_sum = int(t[0].item()), int(t[1].item())
            for pk in list(route_confusion_agg.keys()):
                c = torch.tensor([route_confusion_agg[pk]], device=dev, dtype=torch.float64)
                torch.distributed.all_reduce(c, op=torch.distributed.ReduceOp.SUM)
                route_confusion_agg[pk] = int(c.item())

        route_acc = route_correct_sum / max(route_total_sum, 1)
        stats["route_acc"] = float(route_acc)
        stats["route_correct"] = int(route_correct_sum)
        stats["route_total"] = int(route_total_sum)
        stats["route_confusion"] = dict(route_confusion_agg)

        print(f"[Routing] task={task_type} route_acc={route_acc:.4f} ({route_correct_sum}/{route_total_sum})")
        if route_confusion_agg and route_total_sum > 0:
            print(f"  Routing confusion for GT={task_type}:")
            for pk, cnt in sorted(route_confusion_agg.items(), key=lambda x: -x[1]):
                marker = " <- correct" if pk == task_type else ""
                print(f"    -> {pk}: {cnt}/{route_total_sum} ({100.0 * cnt / route_total_sum:.1f}%){marker}")

    # ── W&B ──
    if wandb_run is not None and task_type is not None:
        task = str(task_type)
        log_epoch = {"epoch": epoch, f"val_epoch/{task}/acc": stats.get("acc")}
        if measure_routing:
            log_epoch[f"val_epoch/{task}/route_acc"] = stats.get("route_acc")
            for pk, cnt in route_confusion_agg.items():
                if route_total_sum > 0:
                    log_epoch[f"val_epoch/{task}/routed_to/{pk}"] = 100.0 * cnt / route_total_sum
        wandb_safe_log(wandb_run, log_epoch, commit=True)
        if global_step is not None:
            log_iter = {"train/step": global_step, f"val_iter/{task}/acc": stats.get("acc")}
            if measure_routing:
                log_iter[f"val_iter/{task}/route_acc"] = stats.get("route_acc")
            wandb_safe_log(wandb_run, log_iter, commit=True)

    return stats
