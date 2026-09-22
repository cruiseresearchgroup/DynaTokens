"""
util/misc.py
============
Distributed-training and logging utilities (adapted from MAE / Flipped-VQA).

SmoothedValue, MetricLogger   running averages with cross-rank synchronisation
init_distributed_mode(args)   sets up torch.distributed from torchrun / PBS env vars
get_world_size / get_rank / is_main_process / save_on_master
NativeScalerWithGradNormCount AMP GradScaler wrapper with `before_step` /
                              `after_step` hooks (used for gradient clipping
                              and the SI importance update)
get_grad_norm_                total gradient norm of a parameter list
"""

import builtins
import datetime
import os
import time
from collections import defaultdict, deque

import torch
import torch.distributed as dist
from torch import inf


class SmoothedValue(object):
    """Track a series of values and provide access to smoothed values over a
    window or the global series average."""

    def __init__(self, window_size=20, fmt=None):
        if fmt is None:
            fmt = "{median:.4f} ({global_avg:.4f})"
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt

    def update(self, value, n=1):
        self.deque.append(value)
        self.count += n
        self.total += value * n

    def synchronize_between_processes(self):
        """Sum count/total across ranks (the deque itself is not synchronised)."""
        if not is_dist_avail_and_initialized():
            return
        use_cuda_tensor = (dist.get_backend() == "nccl") and torch.cuda.is_available()
        device = torch.device(f"cuda:{torch.cuda.current_device()}") if use_cuda_tensor else torch.device("cpu")
        t = torch.tensor([self.count, self.total], dtype=torch.float64, device=device)
        dist.barrier()
        dist.all_reduce(t)
        t = t.tolist()
        self.count = int(t[0])
        self.total = t[1]

    @property
    def median(self):
        return torch.tensor(list(self.deque)).median().item()

    @property
    def avg(self):
        return torch.tensor(list(self.deque), dtype=torch.float32).mean().item()

    @property
    def global_avg(self):
        return self.total / self.count if self.count > 0 else 0.0

    @property
    def max(self):
        return max(self.deque)

    @property
    def value(self):
        return self.deque[-1]

    def __str__(self):
        return self.fmt.format(median=self.median, avg=self.avg, global_avg=self.global_avg,
                               max=self.max, value=self.value)


class MetricLogger(object):
    def __init__(self, delimiter="\t"):
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter

    def update(self, n=1, **kwargs):
        for k, v in kwargs.items():
            if v is None:
                continue
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters[k].update(v, n=n)

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError("'{}' object has no attribute '{}'".format(type(self).__name__, attr))

    def __str__(self):
        return self.delimiter.join(f"{name}: {meter}" for name, meter in self.meters.items())

    def synchronize_between_processes(self):
        for meter in self.meters.values():
            meter.synchronize_between_processes()

    def add_meter(self, name, meter):
        self.meters[name] = meter

    def log_every(self, iterable, print_freq, header=None):
        i = 0
        header = header or ''
        start_time = time.time()
        end = time.time()
        iter_time = SmoothedValue(fmt='{avg:.4f}')
        data_time = SmoothedValue(fmt='{avg:.4f}')
        space_fmt = ':' + str(len(str(len(iterable)))) + 'd'
        log_msg = [header, '[{0' + space_fmt + '}/{1}]', 'eta: {eta}', '{meters}', 'time: {time}', 'data: {data}']
        if torch.cuda.is_available():
            log_msg.append('max mem: {memory:.0f}')
        log_msg = self.delimiter.join(log_msg)
        MB = 1024.0 * 1024.0
        for obj in iterable:
            data_time.update(time.time() - end)
            yield obj
            iter_time.update(time.time() - end)
            if i % print_freq == 0 or i == len(iterable) - 1:
                eta_string = str(datetime.timedelta(seconds=int(iter_time.global_avg * (len(iterable) - i))))
                if torch.cuda.is_available():
                    print(log_msg.format(i, len(iterable), eta=eta_string, meters=str(self), time=str(iter_time),
                                         data=str(data_time), memory=torch.cuda.max_memory_allocated() / MB))
                else:
                    print(log_msg.format(i, len(iterable), eta=eta_string, meters=str(self), time=str(iter_time),
                                         data=str(data_time)))
            i += 1
            end = time.time()
        total_time = time.time() - start_time
        print('{} Total time: {} ({:.4f} s / it)'.format(
            header, str(datetime.timedelta(seconds=int(total_time))), total_time / max(1, len(iterable))))


# ═══════════════════════════════════════════════════════════════════════════════
# Distributed helpers
# ═══════════════════════════════════════════════════════════════════════════════

_builtin_print = builtins.print


def setup_for_distributed(is_master):
    """Disable printing on non-master processes (pass force=True to override)."""
    builtin_print = _builtin_print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False) or (get_world_size() > 8)
        if is_master or force:
            builtin_print('[{}] '.format(datetime.datetime.now().time()), end='')
            builtin_print(*args, **kwargs)

    builtins.print = print


def is_dist_avail_and_initialized():
    return dist.is_available() and dist.is_initialized()


def get_world_size():
    return dist.get_world_size() if is_dist_avail_and_initialized() else 1


def get_rank():
    return dist.get_rank() if is_dist_avail_and_initialized() else 0


def is_main_process():
    return get_rank() == 0


def save_on_master(*args, **kwargs):
    if is_main_process():
        torch.save(*args, **kwargs)


def init_distributed_mode(args):
    """
    Initialise torch.distributed. Rank information is read from the
    environment set by `torchrun` (RANK / WORLD_SIZE / LOCAL_RANK), from
    OpenMPI (with --dist_on_itp) or from SLURM. Without any of these the run
    is single-process.
    """
    if args.dist_on_itp:
        args.rank = int(os.environ['OMPI_COMM_WORLD_RANK'])
        args.world_size = int(os.environ['OMPI_COMM_WORLD_SIZE'])
        args.gpu = int(os.environ['OMPI_COMM_WORLD_LOCAL_RANK'])
        args.dist_url = "tcp://%s:%s" % (os.environ['MASTER_ADDR'], os.environ['MASTER_PORT'])
        os.environ['LOCAL_RANK'] = str(args.gpu)
        os.environ['RANK'] = str(args.rank)
        os.environ['WORLD_SIZE'] = str(args.world_size)
    elif 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.gpu = int(os.environ['LOCAL_RANK'])
    elif 'SLURM_PROCID' in os.environ:
        args.rank = int(os.environ['SLURM_PROCID'])
        args.gpu = args.rank % torch.cuda.device_count()
    else:
        print('Not using distributed mode')
        setup_for_distributed(is_master=True)
        args.distributed = False
        args.gpu = 0
        return

    args.distributed = True
    torch.cuda.set_device(args.gpu)
    args.dist_backend = 'nccl'
    print('| distributed init (rank {}): {}, gpu {}'.format(args.rank, args.dist_url, args.gpu), flush=True)
    torch.distributed.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                         world_size=args.world_size, rank=args.rank,
                                         timeout=datetime.timedelta(seconds=1800))
    torch.distributed.barrier()
    setup_for_distributed(args.rank == 0)


# ═══════════════════════════════════════════════════════════════════════════════
# AMP loss scaler
# ═══════════════════════════════════════════════════════════════════════════════

class NativeScalerWithGradNormCount:
    """
    GradScaler wrapper supporting gradient accumulation and two hooks:
      before_step : called after unscale_ and before optimizer.step
                    (gradients are in true scale, e.g. for clipping)
      after_step  : called right after optimizer.step (parameters have moved)
    """
    state_dict_key = "amp_scaler"

    def __init__(self, device):
        self._scaler = torch.amp.GradScaler(device=device)
        self.device = device

    def __call__(self, loss, optimizer, clip_grad=None, parameters=None, create_graph=False,
                 update_grad=True, before_step=None, after_step=None):
        if hasattr(loss, 'device') and loss.device != self.device:
            loss = loss.to(self.device)
        self._scaler.scale(loss).backward(create_graph=create_graph)
        if not update_grad:
            return None

        self._scaler.unscale_(optimizer)
        if before_step is not None:
            before_step()

        if clip_grad is not None:
            assert parameters is not None
            norm = torch.nn.utils.clip_grad_norm_(parameters, clip_grad)
        else:
            norm = get_grad_norm_(parameters)

        self._scaler.step(optimizer)
        if after_step is not None:
            after_step()
        self._scaler.update()
        return norm

    def state_dict(self):
        return self._scaler.state_dict()

    def load_state_dict(self, state_dict):
        self._scaler.load_state_dict(state_dict)


def get_grad_norm_(parameters, norm_type: float = 2.0) -> torch.Tensor:
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = [p for p in parameters if p.grad is not None]
    norm_type = float(norm_type)
    if len(parameters) == 0:
        return torch.tensor(0.)
    device = parameters[0].grad.device
    if norm_type == inf:
        return max(p.grad.detach().abs().max().to(device) for p in parameters)
    return torch.norm(torch.stack([torch.norm(p.grad.detach(), norm_type).to(device) for p in parameters]), norm_type)
