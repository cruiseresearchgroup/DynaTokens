"""
llama/lookahead_reg.py
======================
LookAhead regularisation of the token generator H_phi (Eq. 16, term L_LA).

    L_LA^t = sum_{tau < t} || H_{phi^m}(z^tau) - H_{phi + delta_phi}(z^tau) ||^2

 * phi^m  : "merged" anchor parameters of the generator (see below);
            treated as a constant.
 * delta_phi : result of a short MAML-style inner loop of the generator on
            the *current* batch, kept differentiable so that the gradient of
            L_LA flows back to phi.
 * z^tau  : past task codes [z_inv, z_sp^tau] taken from the task bank.

Merged anchor phi^m
-------------------
At the end of task t-1 the generator parameters are snapshotted as phi*.
During task t the anchor is updated once per epoch (after
`start_merge_epoch`) as an importance-weighted interpolation between the
previous anchor and the live parameters:

    M   = (1 - norm(I_old)) * norm(I_new)          (top-beta% entries kept)
    a   = M / (norm(I_old) + M)
    phi^m <- (1 - a) * phi^m + a * phi_live

where I_old / I_new are Synaptic-Intelligence (SI) importance scores of the
generator parameters accumulated over past tasks / the current task.

Class
-----
LookaheadRegularizer
    si_begin_task(model)                 reset SI accumulators (start of task)
    accumulate_importance(model)         SI path-integral update (after optimizer.step)
    update_merged_token_generator(model) refresh phi^m (end of epoch)
    start_new_task(model)                finalise I_old, snapshot phi* (end of task)
    compute_reg_loss(model, data, t)     L_LA for the current batch
"""

from contextlib import contextmanager
from typing import Dict, List, Optional

import torch
from torch import nn
from torch.func import functional_call

from llama.dynatokens_components import collect_past_full_codes


def _snapshot_params(module: nn.Module) -> Dict[str, torch.Tensor]:
    """Detached copies of all parameters of `module`, keyed by name."""
    return {k: v.detach().clone() for k, v in module.named_parameters(remove_duplicate=False)}


@contextmanager
def set_train_mode(m: nn.Module, mode: bool):
    prev = m.training
    try:
        m.train(mode)
        yield
    finally:
        m.train(prev)


class LookaheadRegularizer:
    """
    Args
    ----
    inner_steps        : M, number of inner-loop steps (synced from args each epoch)
    inner_lr           : inner-loop step size (synced from the base LR each epoch)
    reg_sample_cap     : max number of past codes used per batch
    enabled            : master switch
    disable_dropout_in_inner : run the generator in eval mode inside the inner loop
    gate_boost_value   : temporarily open the adapter gates during the inner loop
                         so the inner loss depends strongly on the prompts
    mask_topk_percent  : beta, percentage of entries kept in the merge mask
                         (0 or 100 = no sparsification)
    si_xi              : SI damping term xi
    si_clip_neg        : clip negative SI contributions
    merge_quantile     : quantile used to normalise importance scores for merging
    """

    def __init__(
        self,
        inner_steps: int = 2,
        inner_lr: float = 5e-4,
        reg_sample_cap: Optional[int] = 64,
        enabled: bool = True,
        disable_dropout_in_inner: bool = True,
        gate_boost_value: float = 1.5,
        mask_topk_percent: float = 0.0,
        si_xi: float = 1e-3,
        si_clip_neg: bool = True,
        merge_quantile: float = 0.99,
    ):
        self.inner_steps = int(inner_steps)
        self.inner_lr = float(inner_lr)
        self.reg_sample_cap = reg_sample_cap
        self.enabled = bool(enabled)
        self.disable_dropout_in_inner = bool(disable_dropout_in_inner)
        self.gate_boost_value = float(gate_boost_value)
        self.mask_topk_percent = float(mask_topk_percent)
        self.si_xi = float(si_xi)
        self.si_clip_neg = bool(si_clip_neg)
        self.merge_quantile = float(merge_quantile)

        # anchor (phi*) and merged anchor (phi^m)
        self.phi_star: Optional[Dict[str, torch.Tensor]] = None
        self.phi_m: Optional[Dict[str, torch.Tensor]] = None

        # importance scores: I_old (accumulated over past tasks), I_new (current task)
        self.importance_score_phi_star: Optional[Dict[str, torch.Tensor]] = None
        self.importance_score_phi_current: Optional[Dict[str, torch.Tensor]] = None

        # SI state: theta at task start, theta after the last step, path integral w
        self._si_theta_ref: Optional[Dict[str, torch.Tensor]] = None
        self._si_theta_prev: Optional[Dict[str, torch.Tensor]] = None
        self._si_w: Optional[Dict[str, torch.Tensor]] = None

    # ═══════════════════════════════════════════════════════════════════════
    # Synaptic Intelligence (importance scores)
    # ═══════════════════════════════════════════════════════════════════════

    @torch.no_grad()
    def si_begin_task(self, model: nn.Module) -> None:
        """Reset SI accumulators. Call once at the start of every task."""
        theta0 = _snapshot_params(model.token_generator)
        self._si_w = {k: torch.zeros_like(v, dtype=torch.float32) for k, v in theta0.items()}
        self._si_theta_ref = {k: v.to(torch.float32) for k, v in theta0.items()}
        self._si_theta_prev = {k: v.clone().to(torch.float32) for k, v in theta0.items()}
        self.importance_score_phi_current = None

    @torch.no_grad()
    def _si_ensure_init(self, model: nn.Module) -> None:
        if self._si_theta_ref is None or self._si_theta_prev is None or self._si_w is None:
            self.si_begin_task(model)

    @torch.no_grad()
    def _si_refresh_importance_current(self, model: nn.Module) -> None:
        """I_new = relu(w) / ((theta - theta_ref)^2 + xi), stored un-normalised."""
        self._si_ensure_init(model)
        store: Dict[str, torch.Tensor] = {}
        for name, p in model.token_generator.named_parameters(remove_duplicate=False):
            if not p.requires_grad or name not in self._si_theta_ref or name not in self._si_w:
                continue
            theta = p.detach()
            theta_ref = self._si_theta_ref[name].to(device=theta.device, dtype=theta.dtype)
            w = self._si_w[name].to(device=theta.device, dtype=theta.dtype)
            w_eff = torch.relu(w) if self.si_clip_neg else w
            omega = w_eff / ((theta - theta_ref).pow(2) + self.si_xi)
            if not torch.isfinite(omega).all():
                omega = torch.zeros_like(theta)
            store[name] = omega.detach().clone()
        self.importance_score_phi_current = store if store else None

    @torch.no_grad()
    def accumulate_importance(self, model: nn.Module) -> None:
        """
        SI path integral: w_p += (-g_p) * (theta_new - theta_prev).

        Must be called AFTER optimizer.step() and BEFORE the gradients are
        cleared (p.grad is the gradient that was just applied).
        """
        if not self.enabled:
            return
        self._si_ensure_init(model)

        any_update = False
        for name, p in model.token_generator.named_parameters(remove_duplicate=False):
            if not p.requires_grad or p.grad is None:
                continue
            if name not in self._si_theta_prev or name not in self._si_w:
                continue
            theta_new = p.detach().to(torch.float32)
            theta_prev = self._si_theta_prev[name].to(device=theta_new.device)
            g = p.grad.detach().to(device=theta_new.device, dtype=torch.float32)

            contrib = (-g) * (theta_new - theta_prev)
            if not torch.isfinite(contrib).all():
                contrib = torch.zeros_like(contrib)

            self._si_w[name] = self._si_w[name].to(device=theta_new.device) + contrib
            self._si_theta_prev[name] = theta_new.clone()
            any_update = True

        if any_update:
            self._si_refresh_importance_current(model)

    # ═══════════════════════════════════════════════════════════════════════
    # Task lifecycle
    # ═══════════════════════════════════════════════════════════════════════

    @torch.no_grad()
    def start_new_task(self, model: nn.Module):
        """
        Call at the END of task t (before task t+1 starts):
          1. finalise I_new and accumulate I_old <- I_old + I_new
          2. snapshot phi* from the generator, initialise phi^m = phi*
          3. reset SI accumulators for the next task
        """
        if not self.enabled:
            return

        self._si_refresh_importance_current(model)
        if self.importance_score_phi_current is not None:
            if self.importance_score_phi_star is None:
                self.importance_score_phi_star = {
                    k: v.detach().clone() for k, v in self.importance_score_phi_current.items()
                }
            else:
                for k, v in self.importance_score_phi_current.items():
                    if k in self.importance_score_phi_star:
                        self.importance_score_phi_star[k] = self.importance_score_phi_star[k] + v.detach()
                    else:
                        self.importance_score_phi_star[k] = v.detach().clone()
        self.importance_score_phi_current = None

        self.phi_star = _snapshot_params(model.token_generator)
        self.phi_m = {k: v.clone() for k, v in self.phi_star.items()}
        self.si_begin_task(model)

        n_params = sum(v.numel() for v in self.phi_star.values())
        print(f"[LookAhead] snapshotted phi* ({len(self.phi_star)} tensors, {n_params:,} params); "
              f"phi^m initialised; I_old accumulated.")

    # ═══════════════════════════════════════════════════════════════════════
    # Merged anchor phi^m
    # ═══════════════════════════════════════════════════════════════════════

    @torch.no_grad()
    def _quantile_normalise(self, x: torch.Tensor, q: float, eps: float = 1e-8) -> torch.Tensor:
        """clamp(x / quantile(x, q), 0, 1), computed in fp32; x is not modified."""
        if x.numel() == 0:
            return x
        x_f = x.detach().to(torch.float32).view(-1)
        finite = torch.isfinite(x_f)
        if finite.any():
            s = torch.quantile(x_f[finite], q).clamp_min(eps)
        else:
            s = torch.tensor(1.0, device=x.device, dtype=torch.float32)
        return (x.to(torch.float32) / s).clamp(0.0, 1.0).to(dtype=x.dtype)

    @torch.no_grad()
    def update_merged_token_generator(self, model: nn.Module):
        """Refresh phi^m from (phi^m, phi_live, I_old, I_new). Call at the end of an epoch."""
        if not self.enabled or self.phi_star is None:
            return
        if self.phi_m is None:
            self.phi_m = {k: v.clone() for k, v in self.phi_star.items()}
        if self.importance_score_phi_current is None:
            self._si_refresh_importance_current(model)

        I_old_raw = self.importance_score_phi_star
        I_new_raw = self.importance_score_phi_current
        if I_old_raw is None or I_new_raw is None:
            return

        phi_e = _snapshot_params(model.token_generator)
        use_topk = 0.0 < self.mask_topk_percent < 100.0
        frac = self.mask_topk_percent / 100.0 if use_topk else 1.0
        eps = 1e-8
        q = self.merge_quantile

        for name in list(self.phi_m.keys()):
            if name not in phi_e or name not in I_old_raw or name not in I_new_raw:
                continue
            phi_m_param = self.phi_m[name]
            phi_e_param = phi_e[name]

            i_old = self._quantile_normalise(
                I_old_raw[name].to(device=phi_e_param.device, dtype=phi_e_param.dtype), q=q, eps=eps)
            i_new = self._quantile_normalise(
                I_new_raw[name].to(device=phi_e_param.device, dtype=phi_e_param.dtype), q=q, eps=eps)

            score = (1.0 - i_old) * i_new                          # M = (1 - I_old) * I_new

            if use_topk and score.numel() > 0:
                flat = score.view(-1)
                k = max(int(frac * flat.numel()), 1)
                if k < flat.numel():
                    _, idx = torch.topk(flat, k, largest=True, sorted=False)
                    mask_flat = torch.zeros_like(flat, dtype=torch.bool)
                    mask_flat[idx] = True
                    topk_mask = mask_flat.view_as(score)
                else:
                    topk_mask = torch.ones_like(score, dtype=torch.bool)
            else:
                topk_mask = torch.ones_like(score, dtype=torch.bool)

            w_old = i_old
            w_new = torch.where(topk_mask, i_new, torch.zeros_like(i_new))
            denom = w_old + w_new
            alpha_new = torch.where(denom > eps, w_new / (denom + eps),
                                    torch.zeros_like(w_new)).to(phi_e_param.dtype)
            self.phi_m[name] = (1.0 - alpha_new) * phi_m_param + alpha_new * phi_e_param

    # ═══════════════════════════════════════════════════════════════════════
    # Prompt generation with parameter overrides
    # ═══════════════════════════════════════════════════════════════════════

    def _tg_named_buffers(self, model: nn.Module) -> Dict[str, torch.Tensor]:
        return {k: b for k, b in model._tg_wrap.named_buffers(remove_duplicate=False)}

    @contextmanager
    def _temporarily_open_adapter_gates(self, model: nn.Module, val: float):
        """Set gate1 of every adapted layer to `val` inside the context."""
        if not hasattr(model, "adapter_layer") or model.adapter_layer <= 0:
            yield
            return
        layers = model.layers[-model.adapter_layer:]
        saved = [blk.attention.gate1.data.clone() for blk in layers]
        try:
            for blk in layers:
                blk.attention.gate1.data = torch.ones_like(blk.attention.gate1.data) * val
            yield
        finally:
            for blk, old in zip(layers, saved):
                blk.attention.gate1.data.copy_(old)

    def _tg_forward_prompts(
        self,
        model: nn.Module,
        Z: torch.Tensor,
        layer_ids: torch.Tensor,
        param_overrides: Optional[Dict[str, torch.Tensor]],
    ) -> torch.Tensor:
        """P_all = H_{overrides}(Z) via functional_call on the generator wrapper."""
        mapping = self._tg_named_buffers(model)
        if param_overrides is not None:
            mapping.update(param_overrides)
        with set_train_mode(model._tg_wrap, not self.disable_dropout_in_inner):
            return functional_call(model._tg_wrap, mapping, (Z, layer_ids, None, None), {})

    # ═══════════════════════════════════════════════════════════════════════
    # Inner loop (MAML-style, differentiable)
    # ═══════════════════════════════════════════════════════════════════════

    def _build_phi_tilde_init(self, model: nn.Module) -> Dict[str, torch.Tensor]:
        """Copies of phi that stay attached to the graph (no detach)."""
        return {
            f"token_generator.{name}": p.clone().requires_grad_(True)
            for name, p in model.token_generator.named_parameters(remove_duplicate=False)
        }

    def _inner_loss(self, model: nn.Module, data: dict, phi_wrap: Dict[str, torch.Tensor]) -> torch.Tensor:
        """VQA loss on the current batch with prompts generated from phi_wrap."""
        dev = next(model.parameters()).device
        layer_ids = torch.arange(model.adapter_layer, device=dev)
        z = model.z_task_current.full_code_batch()
        with self._temporarily_open_adapter_gates(model, self.gate_boost_value):
            P_all = self._tg_forward_prompts(model, z, layer_ids, param_overrides=phi_wrap)
            out = model(data, inference=False, hp_override={'P_all': P_all})
        return out['vqa_loss'] if isinstance(out, dict) else out

    def _phi_plus_from_inner(self, model: nn.Module, data: dict) -> Dict[str, torch.Tensor]:
        """phi + delta_phi after `inner_steps` gradient steps (create_graph=True)."""
        phi_tilde = self._build_phi_tilde_init(model)
        keys = list(phi_tilde.keys())

        for _ in range(self.inner_steps):
            loss = self._inner_loss(model, data, phi_tilde)
            if not torch.isfinite(loss):
                break
            vars_ = [phi_tilde[k] for k in keys]
            grads = torch.autograd.grad(loss, vars_, create_graph=True, retain_graph=True, allow_unused=True)

            new_phi = {}
            for k, v, g in zip(keys, vars_, grads):
                if g is None or not torch.isfinite(g).all():
                    g = torch.zeros_like(v)
                else:
                    g = g.clamp(-1.0, 1.0)
                new_phi[k] = v - self.inner_lr * g

            if any(not torch.isfinite(new_phi[k]).all() for k in keys):
                break
            phi_tilde = new_phi
        return phi_tilde

    # ═══════════════════════════════════════════════════════════════════════
    # Public: regulariser
    # ═══════════════════════════════════════════════════════════════════════

    def compute_reg_loss(
        self,
        model: nn.Module,
        data: dict,
        t: int,
        past_codes: Optional[List[torch.Tensor]] = None,
    ) -> torch.Tensor:
        """
        L_LA = mean_tau || H_{phi^m}(z^tau) - H_{phi + delta_phi}(z^tau) ||^2

        The anchor term is detached; the look-ahead term keeps the graph so
        the gradient flows back to the live generator parameters.
        """
        dev = next(model.parameters()).device
        zero = lambda: torch.zeros([], device=dev, requires_grad=True)

        if not self.enabled or t <= 1:
            return zero()
        if self.phi_star is None:
            raise RuntimeError("phi_star is None. Call start_new_task(model) at the end of the previous task.")
        if self.phi_m is None:
            self.phi_m = {k: v.clone() for k, v in self.phi_star.items()}

        if past_codes is None:
            bank = getattr(model, "task_bank", None)
            past_codes = [] if (bank is None or len(bank) == 0) else \
                collect_past_full_codes(bank, model.z_task_current.z_inv)
        if len(past_codes) == 0:
            return zero()

        if self.reg_sample_cap is not None and len(past_codes) > self.reg_sample_cap:
            idx = torch.randperm(len(past_codes), device=dev)[: self.reg_sample_cap]
            picked = [past_codes[i.item()] for i in idx]
        else:
            picked = list(past_codes)

        tg_param = next(model.token_generator.parameters())
        Z = torch.stack(picked, dim=0).to(device=tg_param.device, dtype=tg_param.dtype)   # (T, N_z, C_z)
        layer_ids = torch.arange(model.adapter_layer, device=tg_param.device)

        # anchor prompts from phi^m (constant)
        merged_map = {f"token_generator.{k}": v for k, v in self.phi_m.items()}
        with torch.no_grad():
            P_anchor = self._tg_forward_prompts(model, Z, layer_ids, param_overrides=merged_map)
        P_anchor = P_anchor.detach().to(torch.float32)

        # look-ahead prompts from phi + delta_phi (differentiable)
        phi_plus = self._phi_plus_from_inner(model, data)
        P_plus = self._tg_forward_prompts(model, Z, layer_ids, param_overrides=phi_plus).to(torch.float32)

        if not torch.isfinite(P_anchor).all() or not torch.isfinite(P_plus).all():
            return zero()

        diff = P_anchor - P_plus
        reduce_dims = tuple(range(2, diff.ndim)) if diff.ndim >= 3 else (-1,)
        reg = diff.pow(2).mean(dim=reduce_dims).mean()
        if not torch.isfinite(reg):
            return zero()
        return reg
