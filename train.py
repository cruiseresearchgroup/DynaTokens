"""
train.py
========
Continual training of DynaTokens.

Benchmarks (--dataset), see dataloader.BENCHMARKS:
    nextqa          NExT-QA, 8 tasks (one per question type)
    dramaqa         DramaQA, 5 tasks
    nextqa_dramaqa  cross-dataset stream: the 8 NExT-QA tasks followed by the
                    5 DramaQA tasks (task keys "nextqa:TP", ..., "dramaqa:CW")
    visual7w        Visual7W-telling as a single task "visualqa"; its checkpoint
                    can initialise a NExT-QA run (--resume ... --resume_mode init)

Tasks are visited sequentially; after every epoch the model is evaluated on
the validation split of every task seen so far and the average accuracy /
forgetting are reported.

Per task t:
    model.init_task()                         snapshot z_inv*, reset key EMA, re-init z_sp
    epoch 0 ..  epochs-1:
        [epoch 1, t > 1]  warm-start z_sp from the task bank      (Eq. 5-6)
        train_one_epoch                                            (Eq. 16)
        validate on all seen tasks (routing through the task bank at inference)
        [t > 1, epoch >= start_merge_epoch] update merged generator phi^m
    model.end_task()                          push (k^t, z_sp^t) into the bank
    hyper.start_new_task()                    snapshot phi* for the LookAhead term

Outputs written to --output_dir:
    log.txt                    one JSON line per epoch (train + val stats)
    acc_forgetting_stats.csv   per-(train task, eval task, epoch) accuracy / routing / forgetting
    <ckpt_name>_best.pth / _last.pth   with --save_best / --save_last

Resuming (--resume <ckpt>):
    --resume_mode full   restore trainable weights, task code and task bank
                         (continue a stream)
    --resume_mode init   restore only the shared modules (token generator,
                         gates, video projection); task code and bank start
                         fresh (Visual7W -> NExT-QA transfer)

Launch with torchrun (see scripts/run_nextqa.sh and scripts/run_dramaqa.sh):
    torchrun --standalone --nproc_per_node=2 train.py --dataset nextqa ...
"""

import argparse
import csv
import datetime
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist

import util.misc as misc
from dataloader import BENCHMARKS, build_task_list, load_data
from engine import train_one_epoch, val_one_epoch
from llama import LookaheadRegularizer, Tokenizer
from llama_vqa import build_model
from util.misc import NativeScalerWithGradNormCount as NativeScaler

try:
    import wandb
except Exception:
    wandb = None


# Dataset-specific defaults (used when the corresponding flag is not given)
DATASET_DEFAULTS = {
    'nextqa':         dict(adapter_len=10, max_seq_len=128, weight_decay=0.14, bias=3.5),
    'dramaqa':        dict(adapter_len=15, max_seq_len=280, weight_decay=0.10, bias=3.0),
    'nextqa_dramaqa': dict(adapter_len=15, max_seq_len=280, weight_decay=0.12, bias=3.0),
    'visual7w':       dict(adapter_len=10, max_seq_len=128, weight_decay=0.14, bias=3.5),
}


# ═══════════════════════════════════════════════════════════════════════════════
# Arguments
# ═══════════════════════════════════════════════════════════════════════════════

def get_args_parser():
    parser = argparse.ArgumentParser('DynaTokens continual training')

    # ── data ──
    parser.add_argument('--dataset', default='nextqa', choices=list(BENCHMARKS.keys()),
                        help='benchmark / task stream (see dataloader.BENCHMARKS)')
    parser.add_argument('--data_root', default='./data', type=str,
                        help='folder containing <dataset>/split_data and <dataset>/clipvitl14.pth')
    parser.add_argument('--task_order', default=None, type=str,
                        help='comma-separated task keys to select / re-order tasks, e.g. "TP,CW" or '
                             '"nextqa:TP,dramaqa:TW"')
    parser.add_argument('--max_feats', type=int, default=10, help='number of video frames')
    parser.add_argument('--num_workers', default=1, type=int)
    parser.add_argument('--pin_mem', action='store_true')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=True)

    # ── backbone ──
    parser.add_argument('--llama_model_path', default='./checkpoints/Llama-2-7b/', type=str,
                        help='folder with params.json, tokenizer.model and consolidated.*.pth')
    parser.add_argument('--adapter_layer', type=int, default=32, help='number of layers receiving prompts')
    parser.add_argument('--adapter_len', type=int, default=None, help='prompt tokens per layer (dataset default)')
    parser.add_argument('--max_seq_len', type=int, default=None, help='(dataset default)')
    parser.add_argument('--bias', type=float, default=None, help='initial video-attention bias (dataset default)')
    parser.add_argument('--tau', type=float, default=100, help='temperature of the qav loss')

    # ── schedule / optimizer ──
    parser.add_argument('--batch_size', default=2, type=int, help='per-GPU batch size')
    parser.add_argument('--accum_iter', default=1, type=int, help='gradient accumulation steps')
    parser.add_argument('--epochs', default=5, type=int, help='epochs per task')
    parser.add_argument('--warmup_epochs', type=int, default=2)
    parser.add_argument('--start_merge_epoch', default=3, type=int,
                        help='epoch from which the merged generator phi^m is updated')
    parser.add_argument('--weight_decay', type=float, default=None, help='(dataset default)')
    parser.add_argument('--lr', type=float, default=None, help='absolute LR; overrides --blr')
    parser.add_argument('--blr', type=float, default=1e-2, help='base LR: lr = blr * eff_batch / 256')
    parser.add_argument('--min_lr', type=float, default=0.0)
    parser.add_argument('--max_grad_norm', type=float, default=5.0)
    parser.add_argument('--zero_grad_per_step', action='store_true',
                        help='clear gradients after every optimizer step (see engine.py)')

    # ── per-component learning rates ──
    parser.add_argument('--token_gen_lr', type=float, default=1e-3, help='LR of the token generator')
    parser.add_argument('--z_inv_lr', type=float, default=1e-3)
    parser.add_argument('--z_sp_lr', type=float, default=5e-3)

    # ── auxiliary losses (Flipped-VQA) ──
    parser.add_argument('--vaq', action='store_true', help='video+answer -> question loss')
    parser.add_argument('--qav', action='store_true', help='question+answer -> video loss')
    parser.add_argument('--weight_aux_ques_loss', type=float, default=0.8)
    parser.add_argument('--weight_aux_vid_loss', type=float, default=0.1)

    # ── DynaTokens hyper-parameters ──
    parser.add_argument('--alpha', type=float, default=0.5, help='lambda_LA: weight of the LookAhead term')
    parser.add_argument('--looking_head_steps', type=int, default=2, help='M: LookAhead inner-loop steps')
    parser.add_argument('--inner_lr_scale', type=float, default=0.5, help='inner_lr = scale * base LR')
    parser.add_argument('--mask_topk_percent', type=float, default=0.0,
                        help='beta: top-percent of entries kept in the merge mask (0 = no sparsification)')
    parser.add_argument('--lambda_inv', type=float, default=0.25, help='lambda_inv: weight of ||z_inv - z_inv*||^2')
    parser.add_argument('--key_mode', type=str, default='raw', choices=['raw', 'question'],
                        help='retrieval key from question+video (raw) or question only')
    parser.add_argument('--d_k', type=int, default=256, help='retrieval key dimension')
    parser.add_argument('--key_ema_beta', type=float, default=0.99, help='EMA momentum of the task key')
    parser.add_argument('--rho_z', type=float, default=0.1, help='softmax temperature of the warm-start')

    # ── inference / ablations ──
    parser.add_argument('--use_task_id', action='store_true',
                        help='oracle routing: use the ground-truth task id instead of key retrieval')
    parser.add_argument('--code_mode', type=str, default='retrieved',
                        choices=['retrieved', 'dissimilar', 'random', 'zsp_zero', 'zinv_zero'],
                        help='code-swap ablation at inference (see TaskBank)')

    # ── checkpoints ──
    parser.add_argument('--output_dir', default='./outputs/debug')
    parser.add_argument('--resume', default='', help='checkpoint (.pth) saved by this script')
    parser.add_argument('--resume_mode', default='full', choices=['full', 'init'],
                        help='full: weights + task code + task bank; init: shared modules only '
                             '(token generator, gates, video projection), used for Visual7W -> NExT-QA')
    parser.add_argument('--save_best', action='store_true', help='save the checkpoint with the best avg val acc')
    parser.add_argument('--save_last', action='store_true', help='save a checkpoint after the last task')
    parser.add_argument('--ckpt_name', type=str, default='dynatokens')

    # ── misc ──
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://')
    parser.add_argument('--use_wandb', action='store_true')
    parser.add_argument('--wandb_project', type=str, default='DynaTokens')
    parser.add_argument('--wandb_name', type=str, default=None)
    parser.add_argument('--wandb_log_freq', type=int, default=50)
    return parser


def apply_dataset_defaults(args):
    for k, v in DATASET_DEFAULTS[args.dataset].items():
        if getattr(args, k) is None:
            setattr(args, k, v)
    return args


# ═══════════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════════

def get_group_lr(optimizer, group_name: str, default=None):
    for g in optimizer.param_groups:
        if g.get('name') == group_name:
            return float(g['lr'])
    return default


def build_optimizer(args, model):
    """
    AdamW with one param group per component (token generator / z_inv / z_sp /
    gates+projections), each split into decay / no-decay (bias, norm) subsets.
    """
    # no weight decay for 1-D tensors (norm weights) and biases (same rule as timm)
    no_decay_ids = {
        id(p) for n, p in model.named_parameters()
        if p.requires_grad and (p.ndim <= 1 or n.endswith(".bias"))
    }

    def split(params):
        no_decay = [p for p in params if id(p) in no_decay_ids]
        decay = [p for p in params if id(p) not in no_decay_ids]
        return no_decay, decay

    token_gen_params = list(model.token_generator.parameters())
    z_inv_params = [model.z_task_current.z_inv]
    z_sp_params = [model.z_task_current.z_sp]
    named_ids = {id(p) for p in token_gen_params + z_inv_params + z_sp_params}
    other_params = [p for p in model.parameters() if p.requires_grad and id(p) not in named_ids]

    tg_no_decay, tg_decay = split(token_gen_params)
    other_no_decay, other_decay = split(other_params)

    def group(params, lr, wd, name):
        return {'params': params, 'lr': lr, 'initial_lr': lr, 'weight_decay': wd, 'name': name}

    param_groups = [
        group(tg_no_decay, args.token_gen_lr, 0.0, 'token_generator_no_decay'),
        group(tg_decay, args.token_gen_lr, args.weight_decay, 'token_generator_decay'),
        group(z_inv_params, args.z_inv_lr, 0.0, 'z_inv'),
        group(z_sp_params, args.z_sp_lr, 0.0, 'z_sp'),
        group(other_no_decay, args.lr, 0.0, 'other_no_decay'),
        group(other_decay, args.lr, args.weight_decay, 'other_decay'),
    ]
    param_groups = [g for g in param_groups if len(g['params']) > 0]
    for g in param_groups:
        n = sum(p.numel() for p in g['params'])
        print(f"  {g['name']:>26}: {len(g['params']):4d} tensors {n:>12,} params  lr={g['lr']:.2e} wd={g['weight_decay']}")
    return torch.optim.AdamW(param_groups, betas=(0.9, 0.95))


def save_checkpoint(args, model, path, epoch, task_key=None, avg_acc=None):
    """Trainable parameters + DynaTokens buffers + task bank (the frozen backbone is not saved)."""
    trainable = {n for n, p in model.named_parameters() if p.requires_grad}
    state = {}
    for k, v in model.state_dict().items():
        if k in trainable or k.startswith('retrieval_key_module.') or k == 'z_inv_snapshot':
            state[k] = v.detach().cpu()
    torch.save({
        'model': state,
        'task_bank': model.task_bank.state_dict(),
        'epoch': epoch,
        'task_key': task_key,
        'avg_acc': avg_acc,
        'args': vars(args),
    }, path)
    print(f"[CKPT] saved {path} ({len(state)} tensors, bank={list(model.task_bank.bank.keys())})")


# state-dict keys that belong to the task in progress / the stream (not loaded in 'init' mode)
TASK_STATE_PREFIXES = ('z_task_current.', 'z_inv_snapshot', 'retrieval_key_module.')


def load_checkpoint(model, path, mode='full'):
    """
    mode='full': trainable weights, DynaTokens buffers and task bank.
    mode='init': shared modules only (token generator, gates, temporal_emb,
                 visual_proj); task code, key EMA and task bank stay fresh.
    """
    ckpt = torch.load(path, map_location='cpu', weights_only=False)
    state = ckpt['model']
    if mode == 'init':
        state = {k: v for k, v in state.items() if not k.startswith(TASK_STATE_PREFIXES)}
    model_sd = model.state_dict()
    shape_ok = {k: v for k, v in state.items() if k in model_sd and tuple(v.shape) == tuple(model_sd[k].shape)}
    _, unexpected = model.load_state_dict(shape_ok, strict=False)
    if mode == 'full':
        model.task_bank.load_state_dict(ckpt.get('task_bank'))
    print(f"[CKPT] loaded {path} (mode={mode}): {len(shape_ok)}/{len(state)} tensors, "
          f"{len(unexpected)} unexpected, bank={model.task_bank.all_task_ids()}")


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main(args):
    misc.init_distributed_mode(args)
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    if torch.cuda.is_available() and str(args.device).startswith("cuda"):
        device = torch.device(f"cuda:{args.gpu}")
    else:
        device = torch.device(args.device)

    print("{}".format(args).replace(', ', ',\n'))

    # ── CSV log ──
    csv_fieldnames = ['epoch', 'train_task_index', 'train_task_type', 'eval_task_index', 'eval_task_type',
                      'val_acc', 'route_acc', 'average_acc', 'forgetting', 'forgetting_this_task']
    csv_path = os.path.join(args.output_dir, "acc_forgetting_stats.csv")
    if misc.is_main_process():
        with open(csv_path, mode="w", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, fieldnames=csv_fieldnames).writeheader()

    # ── W&B ──
    wandb_run = None
    if args.use_wandb and wandb is not None and misc.is_main_process():
        wandb_run = wandb.init(project=args.wandb_project, name=args.wandb_name, config=vars(args))
        wandb.define_metric("train/step")
        wandb.define_metric("epoch")
        for pattern, step in [("train/*", "train/step"), ("train_epoch/*", "epoch"), ("val_epoch/*", "epoch"),
                              ("val_iter/*", "train/step"), ("cl/*", "epoch")]:
            wandb.define_metric(pattern, step_metric=step)

    # ── seed ──
    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    cudnn.benchmark = False
    cudnn.deterministic = True

    # ── model ──
    tokenizer = Tokenizer(model_path=f'{args.llama_model_path}/tokenizer.model')
    model = build_model(args)
    model.to(device)
    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.gpu], output_device=args.gpu,
            broadcast_buffers=False, find_unused_parameters=False, gradient_as_bucket_view=True,
        )
        model_without_ddp = model.module

    # ── optimizer ──
    eff_batch_size = args.batch_size * args.accum_iter * misc.get_world_size()
    if args.lr is None:
        args.lr = args.blr * eff_batch_size / 256
    print(f"\nactual lr: {args.lr:.2e}, effective batch size: {eff_batch_size}")
    optimizer = build_optimizer(args, model_without_ddp)
    loss_scaler = NativeScaler(device=device)

    # ── LookAhead regulariser ──
    hyper = LookaheadRegularizer(
        inner_steps=args.looking_head_steps,
        inner_lr=5e-5,
        reg_sample_cap=64,
        enabled=True,
        mask_topk_percent=args.mask_topk_percent,
    )

    # ── resume ──
    task_offset = 0                                  # tasks already in the bank before this run
    if args.resume:
        load_checkpoint(model_without_ddp, args.resume, mode=args.resume_mode)
        task_offset = len(model_without_ddp.task_bank)
        if task_offset > 0:
            # continue a stream: the loaded generator becomes the LookAhead anchor phi*
            hyper.start_new_task(model_without_ddp)

    # ── task order ──
    tasks = build_task_list(args.dataset, args.task_order)
    print(f"\nTask stream ({args.dataset}): {[t.key for t in tasks]}")

    processed = []                                   # TaskSpecs seen so far
    acc_matrix = np.zeros((len(tasks), len(tasks)))
    global_step = 0
    best_avg_acc = -1.0
    start_time = time.time()

    for task in tasks:
        qtype = task.key
        print(f'\n{"=" * 80}\nTraining task: {qtype}  (dataset={task.dataset}, type={task.qtype})\n{"=" * 80}')
        processed.append(task)
        t_index = task_offset + len(processed)      # 1-based index in the whole stream

        model_without_ddp.init_task(question_type=qtype, task_index=t_index)
        hyper.si_begin_task(model_without_ddp)
        warm_started = False

        for epoch in range(args.epochs):
            print(f'\n--- Task {qtype} | Epoch {epoch} ---')

            # warm-start z_sp after the first epoch of every task but the first (Eq. 5-6)
            if not warm_started and epoch > 0 and t_index > 1 and len(model_without_ddp.task_bank) > 0:
                model_without_ddp.warm_start_z_sp_from_data()
                warm_started = True

            # inner-loop LR of the LookAhead term follows the base LR
            hyper.inner_lr = float(args.inner_lr_scale * get_group_lr(optimizer, 'other_decay', default=args.lr))
            hyper.inner_steps = args.looking_head_steps

            data_loader_train = load_data(args, tokenizer, split='train', task=task)
            if args.distributed:
                data_loader_train.sampler.set_epoch(epoch)

            train_stats, global_step = train_one_epoch(
                model, data_loader_train, optimizer, epoch, loss_scaler,
                args=args, current_qtype=qtype, hyper=hyper, task_index=t_index,
                wandb_run=wandb_run, global_step=global_step, wandb_log_freq=args.wandb_log_freq,
            )
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            # ── validation on all seen tasks (rank 0, full validation split) ──
            if misc.is_main_process():
                model_without_ddp.push_current_to_bank_temp(question_type=qtype)
                print(f"\n  [Validation] task bank: {model_without_ddp.task_bank.all_task_ids()}")
                do_routing = not args.use_task_id

                epoch_records = []
                for i, val_task in enumerate(processed):
                    val_type = val_task.key
                    data_loader_val = load_data(args, tokenizer, split='val', task=val_task, distributed=False)
                    val_stats = val_one_epoch(
                        model_without_ddp, data_loader_val, epoch, args=args, task_type=val_type,
                        wandb_run=wandb_run, global_step=global_step, measure_routing=do_routing,
                    )
                    acc = float(val_stats['acc'])
                    route_acc = float(val_stats.get('route_acc', -1.0))
                    acc_matrix[i, len(processed) - 1] = acc
                    msg = f"  {val_type}: acc={acc:.4f}"
                    if do_routing:
                        msg += f"  route_acc={route_acc:.4f} ({val_stats.get('route_correct', 0)}/{val_stats.get('route_total', 0)})"
                    print(msg)
                    epoch_records.append({
                        'epoch': epoch, 'train_task_index': t_index, 'train_task_type': qtype,
                        'eval_task_index': i + 1, 'eval_task_type': val_type,
                        'val_acc': acc, 'route_acc': route_acc if do_routing else None,
                    })
                model_without_ddp.pop_current_from_bank_temp()

                # average accuracy / forgetting over seen tasks
                average_acc = float(np.mean([r['val_acc'] for r in epoch_records]))
                route_accs = [r['route_acc'] for r in epoch_records if r['route_acc'] is not None and r['route_acc'] >= 0]
                avg_route_acc = float(np.mean(route_accs)) if route_accs else -1.0

                n_seen = len(processed)               # tasks of this run (rows/cols of acc_matrix)
                cur_col = n_seen - 1
                max_up_to_now = np.max(acc_matrix[:n_seen, :n_seen], axis=1)
                per_task_forgetting = max_up_to_now - acc_matrix[:n_seen, cur_col]
                per_task_forgetting[cur_col] = 0.0
                forgetting_avg = float(per_task_forgetting[:cur_col].mean()) if n_seen > 1 else 0.0

                print(f"\n  Average accuracy: {average_acc:.4f}   Avg forgetting: {forgetting_avg:.4f}"
                      + (f"   Avg routing acc: {avg_route_acc:.4f}" if route_accs else ""))

                for i, rec in enumerate(epoch_records):
                    rec['average_acc'] = average_acc
                    rec['forgetting'] = forgetting_avg
                    rec['forgetting_this_task'] = float(per_task_forgetting[i])
                with open(csv_path, mode="a", newline="", encoding="utf-8") as f:
                    csv.DictWriter(f, fieldnames=csv_fieldnames).writerows(epoch_records)

                if wandb_run is not None:
                    logd = {"epoch": epoch, "cl/average_acc": average_acc,
                            "cl/forgetting_avg": forgetting_avg, "cl/train_task_index": t_index}
                    if avg_route_acc >= 0:
                        logd["cl/avg_route_acc"] = avg_route_acc
                    for j, tname in enumerate(t.key for t in processed):
                        logd[f"cl/acc/{tname}"] = float(acc_matrix[j, cur_col])
                        logd[f"cl/forgetting/{tname}"] = float(per_task_forgetting[j])
                        ra = epoch_records[j]['route_acc']
                        if ra is not None and ra >= 0:
                            logd[f"cl/route_acc/{tname}"] = float(ra)
                    wandb_run.log(logd, commit=True)

                log_stats = {**{f'train_{k}': v for k, v in train_stats.items()}, 'epoch': epoch,
                             'task': qtype, 'average_acc': average_acc, 'forgetting': forgetting_avg,
                             **{f'val_acc/{r["eval_task_type"]}': r['val_acc'] for r in epoch_records}}
                with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                    f.write(json.dumps(log_stats) + "\n")

                if args.save_best and average_acc > best_avg_acc:
                    best_avg_acc = average_acc
                    model_without_ddp.push_current_to_bank_temp(question_type=qtype)
                    save_checkpoint(args, model_without_ddp, os.path.join(args.output_dir, f"{args.ckpt_name}_best.pth"),
                                    epoch=epoch, task_key=qtype, avg_acc=average_acc)
                    model_without_ddp.pop_current_from_bank_temp()

            if args.distributed and dist.is_initialized():
                dist.barrier()

            # update the merged generator phi^m used as LookAhead anchor
            if t_index > 1 and epoch >= args.start_merge_epoch:
                hyper.update_merged_token_generator(model_without_ddp)

        model_without_ddp.end_task(question_type=qtype)
        hyper.start_new_task(model_without_ddp)
        print(f'\nCompleted task: {qtype}')

    if args.save_last and misc.is_main_process():
        save_checkpoint(args, model_without_ddp, os.path.join(args.output_dir, f"{args.ckpt_name}_last.pth"),
                        epoch=args.epochs - 1, task_key=tasks[-1].key)

    total_time = time.time() - start_time
    print(f'\n{"=" * 80}\nTraining completed in {datetime.timedelta(seconds=int(total_time))}')
    if args.save_best:
        print(f'Best average accuracy: {best_avg_acc:.4f}')
    print(f'Final accuracy matrix (rows: eval task, cols: after training task):\n{np.round(acc_matrix, 4)}')
    print("=" * 80)

    if wandb_run is not None:
        wandb_run.finish()
    if args.distributed and dist.is_initialized():
        dist.destroy_process_group()


if __name__ == '__main__':
    args = apply_dataset_defaults(get_args_parser().parse_args())
    main(args)
