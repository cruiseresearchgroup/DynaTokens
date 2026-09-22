"""
llama/dynatokens_components.py
==============================
Building blocks of the DynaTokens task-code machinery (everything that is
*not* the frozen LLaMA backbone and *not* the token generator H_phi).

Components
----------
DecomposedTaskCode
    The task code z^t = [z_inv, z_sp^t] (Eq. 1). z_inv is shared across all
    tasks; z_sp^t is task-specific and re-initialised for every new task.

warm_start_z_sp
    Warm-start of z_sp^t from the task bank (Eq. 5-6): a softmax-weighted
    combination of past z_sp^tau, weighted by key similarity, plus noise.

make_rademacher_projection
    Deterministic +-1/sqrt(d_k) random projection used to build retrieval
    keys from frozen features (identical on every rank / restart).

RetrievalKeyModule
    Builds a per-sample retrieval key r_i from frozen features and keeps an
    EMA of those keys as the task key k^t (Eq. 8-11). Two modes:
      'raw'      : key from question embedding + raw video feature
      'question' : key from question embedding only

TaskBank
    Stores (k^t, z_sp^t) pairs (Eq. 13) and retrieves the best matching task
    for a batch of test keys by cosine similarity (Eq. 14-15).

z_inv_reg_loss
    ||z_inv - z_inv*||^2, the last term of Eq. 16.

collect_past_full_codes
    Builds [z_inv, z_sp^tau] for every past task; used by the LookAhead
    regulariser.
"""

from __future__ import annotations

import math
from collections import OrderedDict
from typing import Dict, List, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


def normalize_bank_key(x):
    """Convert a task identifier (str / int / scalar tensor) to a hashable key."""
    if x is None:
        return None
    if isinstance(x, (str, int)):
        return x
    if torch.is_tensor(x):
        if x.numel() != 1:
            raise ValueError(f"Expected a scalar tensor task key, got shape={tuple(x.shape)}")
        return x.item()
    return x


# ═══════════════════════════════════════════════════════════════════════════════
# 1. Decomposed task code  z^t = [z_inv, z_sp^t]   (Eq. 1)
# ═══════════════════════════════════════════════════════════════════════════════

class DecomposedTaskCode(nn.Module):
    """
    z^t = [z_inv, z_sp^t] in R^{N_z x C_z}

    z_inv : shared across tasks (N_inv tokens), regularised towards its snapshot.
    z_sp  : task-specific (N_sp tokens), re-initialised / warm-started per task.
    """

    def __init__(self, n_inv: int, n_sp: int, c_z: int, init_std: float = 0.02):
        super().__init__()
        self.n_inv = n_inv
        self.n_sp = n_sp
        self.n_z = n_inv + n_sp
        self.c_z = c_z

        self.z_inv = nn.Parameter(torch.randn(n_inv, c_z))
        nn.init.trunc_normal_(self.z_inv, std=init_std)

        self.z_sp = nn.Parameter(torch.randn(n_sp, c_z))
        nn.init.trunc_normal_(self.z_sp, std=init_std)

    def full_code(self) -> torch.Tensor:
        """z^t = [z_inv, z_sp^t] -> (N_z, C_z)"""
        return torch.cat([self.z_inv, self.z_sp], dim=0)

    def full_code_batch(self) -> torch.Tensor:
        """(1, N_z, C_z), ready to be fed to the token generator."""
        return self.full_code().unsqueeze(0)

    @torch.no_grad()
    def snapshot_z_inv(self) -> torch.Tensor:
        """Detached copy of z_inv (z_inv*), taken before training a new task."""
        return self.z_inv.detach().clone()

    @torch.no_grad()
    def export_z_sp(self) -> torch.Tensor:
        """Detached copy of z_sp for storing in the task bank."""
        return self.z_sp.detach().clone()

    @torch.no_grad()
    def reset_z_sp(self, std: float = 0.02):
        """Random re-initialisation of z_sp (used at the start of every task)."""
        nn.init.trunc_normal_(self.z_sp, std=std)

    @torch.no_grad()
    def set_z_sp(self, z_sp_new: torch.Tensor):
        """Overwrite z_sp in place (used by warm_start_z_sp)."""
        assert z_sp_new.shape == self.z_sp.shape, (
            f"Shape mismatch: got {z_sp_new.shape}, expected {self.z_sp.shape}"
        )
        self.z_sp.copy_(z_sp_new)


# ═══════════════════════════════════════════════════════════════════════════════
# 2. Warm-start of z_sp   (Eq. 5-6)
# ═══════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def warm_start_z_sp(
    task_code: DecomposedTaskCode,
    task_bank: "TaskBank",
    k_0_t: torch.Tensor,
    rho_z: float = 0.1,
    noise_std: float = 0.02,
):
    """
    z_sp^t = sum_{tau<t} w_tau * z_sp^tau + eps,
    w_tau  = softmax( cos(k_0^t, k^tau) / rho_z ),
    eps    ~ N(0, noise_std^2 I)
    """
    if len(task_bank) == 0:
        task_code.reset_z_sp(std=noise_std)
        return

    device = k_0_t.device
    keys_past = task_bank.all_keys_tensor(device=device)      # (T, D_k)
    z_sps_past = task_bank.all_z_sp_tensor(device=device)     # (T, N_sp, C_z)

    k_0_t_norm = F.normalize(k_0_t.float().unsqueeze(0), dim=-1)   # (1, D_k)
    keys_norm = F.normalize(keys_past.float(), dim=-1)              # (T, D_k)
    cos_sim = (k_0_t_norm @ keys_norm.T).squeeze(0)                # (T,)

    w = torch.softmax(cos_sim / rho_z, dim=0)                      # (T,)
    z_sp_init = torch.einsum("t,tnc->nc", w, z_sps_past.float())   # (N_sp, C_z)
    z_sp_init = z_sp_init + torch.randn_like(z_sp_init) * noise_std

    task_code.set_z_sp(z_sp_init.to(task_code.z_sp.dtype))


# ═══════════════════════════════════════════════════════════════════════════════
# 3. Rademacher random projection (used for retrieval keys)
# ═══════════════════════════════════════════════════════════════════════════════

def make_rademacher_projection(
    row_dim: int,
    proj_dim: int,
    seed: int,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """
    Random projection matrix (proj_dim, row_dim) with entries +-1/sqrt(proj_dim).
    Seeded, hence identical across ranks and restarts; approximately distance
    preserving (Johnson-Lindenstrauss).
    """
    Pi = torch.empty(proj_dim, row_dim, device=device, dtype=dtype)
    gen = torch.Generator(device=device)
    gen.manual_seed(int(seed))
    Pi.bernoulli_(0.5, generator=gen).mul_(2.0).sub_(1.0).mul_(1.0 / math.sqrt(proj_dim))
    return Pi


# ═══════════════════════════════════════════════════════════════════════════════
# 4. Retrieval keys   (Eq. 8-11)
# ═══════════════════════════════════════════════════════════════════════════════

class RetrievalKeyModule(nn.Module):
    """
    Foundation-guided retrieval keys computed from *frozen* features, so they
    never drift while the adapters are trained.

    mode='raw'      : r_i = Norm( Pi_q * Pool(question_emb) + Pi_v * Pool(video_raw) )
    mode='question' : r_i = Norm( Pi_q * Pool(question_emb) )

    During training an EMA of the per-batch mean key is kept (mu); the task
    key is k^t = Norm(mu) (see get_task_key). All state lives in buffers, the
    module has no trainable parameters.
    """

    def __init__(self, d_llm: int, d_k: int, beta: float = 0.99,
                 mode: str = 'raw', d_video: int = 768):
        super().__init__()
        if mode not in ('raw', 'question'):
            raise ValueError(f"Unknown key mode: {mode}. Use 'raw' or 'question'.")
        self.mode = mode
        self.d_k = d_k
        self.beta = beta

        cpu = torch.device("cpu")
        self.register_buffer("Pi_q", make_rademacher_projection(d_llm, d_k, seed=42, device=cpu))
        if mode == 'raw':
            self.register_buffer("Pi_v", make_rademacher_projection(d_video, d_k, seed=137, device=cpu))
        self.register_buffer("mu", torch.zeros(d_k))
        self.register_buffer("_ready", torch.tensor(False))

    # ── per-sample keys ──

    def compute_sample_key_raw(self, question_features: torch.Tensor,
                               video_features: torch.Tensor) -> torch.Tensor:
        """(B, Lq, D_llm) + (B, M, d_video) -> (B, D_k)"""
        q_pooled = question_features.float().mean(dim=1)
        v_pooled = video_features.float().mean(dim=1)
        combined = F.linear(q_pooled, self.Pi_q.float()) + F.linear(v_pooled, self.Pi_v.float())
        return F.normalize(combined, dim=-1)

    def compute_sample_key_question(self, question_features: torch.Tensor) -> torch.Tensor:
        """(B, Lq, D_llm) -> (B, D_k)"""
        q_pooled = question_features.float().mean(dim=1)
        return F.normalize(F.linear(q_pooled, self.Pi_q.float()), dim=-1)

    def compute_sample_keys(self, question_features: torch.Tensor,
                            video_features: torch.Tensor = None) -> torch.Tensor:
        """Dispatch on `mode`."""
        if self.mode == 'raw':
            return self.compute_sample_key_raw(question_features, video_features)
        return self.compute_sample_key_question(question_features)

    # ── EMA task key ──

    @torch.no_grad()
    def update_ema(self, sample_keys: torch.Tensor):
        """mu <- beta * mu + (1 - beta) * mean_i(r_i); first call initialises mu."""
        batch_mean = sample_keys.float().mean(dim=0)
        if not bool(self._ready.item()):
            self.mu.copy_(batch_mean)
            self._ready.fill_(True)
        else:
            self.mu.mul_(self.beta).add_(batch_mean, alpha=1 - self.beta)

    @torch.no_grad()
    def get_task_key(self) -> torch.Tensor:
        """k^t = Norm(mu)"""
        return F.normalize(self.mu.float(), dim=0)

    @torch.no_grad()
    def reset_ema(self):
        """Called at the start of every task."""
        self.mu.zero_()
        self._ready.fill_(False)


# ═══════════════════════════════════════════════════════════════════════════════
# 5. Task bank   (Eq. 13-15)
# ═══════════════════════════════════════════════════════════════════════════════

class TaskBank:
    """
    B_t = B_{t-1} U {(k^t, z_sp^t)}   stored on CPU in fp32.

    Retrieval: t* = argmax_{tau<=t} cos(r_test, k^tau)

    `code_mode` (default 'retrieved') is an ablation switch used for the
    routing analysis in the paper:
      'retrieved' : normal nearest-key retrieval
      'dissimilar': retrieve the *least* similar task instead (argmin)
    """

    def __init__(self, max_size: int = 256):
        self.max_size = max_size
        self.bank: "OrderedDict[str, Dict[str, torch.Tensor]]" = OrderedDict()
        self.code_mode = 'retrieved'

    def __len__(self) -> int:
        return len(self.bank)

    @torch.no_grad()
    def push(self, task_id: str, key: torch.Tensor, z_sp: torch.Tensor):
        """Insert or overwrite the entry for `task_id`; evicts oldest if full."""
        entry = {
            "key": key.detach().float().cpu(),
            "z_sp": z_sp.detach().float().cpu(),
        }
        if task_id in self.bank:
            del self.bank[task_id]
        self.bank[task_id] = entry
        while len(self.bank) > self.max_size:
            self.bank.popitem(last=False)

    def remove(self, task_id: str):
        self.bank.pop(task_id, None)

    def all_keys_tensor(self, device="cpu") -> torch.Tensor:
        return torch.stack([e["key"] for e in self.bank.values()], dim=0).to(device)

    def all_z_sp_tensor(self, device="cpu") -> torch.Tensor:
        return torch.stack([e["z_sp"] for e in self.bank.values()], dim=0).to(device)

    def all_task_ids(self) -> List[str]:
        return list(self.bank.keys())

    def get_past_z_sp_list(self) -> List[torch.Tensor]:
        return [e["z_sp"].clone() for e in self.bank.values()]

    @torch.no_grad()
    def retrieve_batch(self, r_test: torch.Tensor, z_inv: torch.Tensor) -> Tuple[List[str], torch.Tensor]:
        """
        r_test: (B, D_k) test keys; z_inv: (N_inv, C_z).
        Returns the predicted task id per sample and the full codes
        [z_inv, z_sp^{t*}] of shape (B, N_z, C_z).
        """
        assert len(self.bank) > 0, "TaskBank is empty"
        B = r_test.shape[0]
        device = r_test.device
        keys = self.all_keys_tensor(device=device)

        r_norm = F.normalize(r_test.float(), dim=-1)
        k_norm = F.normalize(keys.float(), dim=-1)
        cos_sim = r_norm @ k_norm.T                                   # (B, T)

        if self.code_mode == 'dissimilar':
            best_indices = cos_sim.argmin(dim=1).tolist()
        else:
            best_indices = cos_sim.argmax(dim=1).tolist()

        task_ids = self.all_task_ids()
        pred_task_ids = [task_ids[i] for i in best_indices]
        best_z_sps = self.all_z_sp_tensor(device=device)[best_indices]  # (B, N_sp, C_z)

        z_inv_expanded = z_inv.unsqueeze(0).expand(B, -1, -1).to(device)
        z_full = torch.cat([z_inv_expanded, best_z_sps], dim=1)
        return pred_task_ids, z_full

    # ── (de)serialisation for checkpoints ──

    def state_dict(self) -> "OrderedDict[str, Dict[str, torch.Tensor]]":
        return OrderedDict(
            (k, {kk: vv.detach().cpu() for kk, vv in v.items()}) for k, v in self.bank.items()
        )

    def load_state_dict(self, packed):
        self.bank.clear()
        if not packed:
            return
        for k, v in packed.items():
            self.push(k, v["key"], v["z_sp"])


# ═══════════════════════════════════════════════════════════════════════════════
# 6. Regularisation helpers   (Eq. 16)
# ═══════════════════════════════════════════════════════════════════════════════

def z_inv_reg_loss(z_inv_current: torch.Tensor, z_inv_snapshot: torch.Tensor) -> torch.Tensor:
    """||z_inv - z_inv*||^2 (last term of Eq. 16)."""
    return (z_inv_current.float() - z_inv_snapshot.float().detach()).pow(2).sum()


def collect_past_full_codes(task_bank: TaskBank, z_inv: torch.Tensor) -> List[torch.Tensor]:
    """[z_inv, z_sp^tau] for every past task tau in the bank (for LookAhead)."""
    full_codes = []
    for z_sp_tau in task_bank.get_past_z_sp_list():
        full_codes.append(torch.cat([z_inv.detach(), z_sp_tau.to(z_inv.device)], dim=0))
    return full_codes
