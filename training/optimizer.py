"""
training/optimizer.py
─────────────────────
Adafactor optimizer (Shazeer & Stern 2018) — the standard optimizer for T5.

Key properties vs Adam:
  - Factored second-moment estimation   → much lower memory footprint
  - No first moment (momentum)          → even less memory
  - Scale-invariant LR                  → robust to weight magnitudes
  - Clips update norm                   → implicit gradient clipping

This is a clean, standalone re-implementation matching the original paper
and the Mesh TensorFlow reference implementation used to train T5.
"""

import math
import torch
from torch.optim import Optimizer
from typing import List, Optional


class Adafactor(Optimizer):
    """
    Adafactor optimizer.

    Args:
        params              : model parameters
        lr                  : explicit learning rate (required when relative_step=False)
        eps                 : (ε1, ε2)  regularisation constants
        clip_threshold      : RMS clip of parameter updates  (ρ in paper)
        decay_rate          : coefficient for factored second moment  (β in paper; -0.8 → schedule)
        beta1               : optional first moment coefficient (None = no momentum, saves memory)
        weight_decay        : L2 regularization
        scale_parameter     : if True, scale lr by rms(parameter)
        relative_step       : if True, ignore `lr` and use internal schedule
        warmup_init         : initialise with small lr during first steps
    """

    def __init__(
        self,
        params,
        lr:              Optional[float] = None,
        eps:             tuple           = (1e-30, 1e-3),
        clip_threshold:  float           = 1.0,
        decay_rate:      float           = -0.8,
        beta1:           Optional[float] = None,
        weight_decay:    float           = 0.0,
        scale_parameter: bool            = True,
        relative_step:   bool            = True,
        warmup_init:     bool            = False,
    ):
        if lr is not None and relative_step:
            raise ValueError("Cannot use explicit lr with relative_step=True")
        if lr is None and not relative_step:
            raise ValueError("Must supply lr when relative_step=False")

        defaults = dict(
            lr              = lr,
            eps             = eps,
            clip_threshold  = clip_threshold,
            decay_rate      = decay_rate,
            beta1           = beta1,
            weight_decay    = weight_decay,
            scale_parameter = scale_parameter,
            relative_step   = relative_step,
            warmup_init     = warmup_init,
        )
        super().__init__(params, defaults)

    @staticmethod
    def _get_lr(param_group: dict, param_state: dict) -> float:
        if param_group["relative_step"]:
            min_step = 1e-6 if param_group["warmup_init"] else 1e-2
            rel_step = min(min_step, 1.0 / math.sqrt(param_state["step"]))
            return rel_step * param_state.get("rms", 1.0)
        return param_group["lr"]

    @staticmethod
    def _rms(tensor: torch.Tensor) -> float:
        return tensor.norm(2).item() / (tensor.numel() ** 0.5)

    @staticmethod
    def _approx_sq_grad(exp_avg_sq_row: torch.Tensor,
                         exp_avg_sq_col: torch.Tensor) -> torch.Tensor:
        """Reconstruct full 2-D second moment from factored row/col vectors."""
        r_factor = (exp_avg_sq_row / exp_avg_sq_row.mean(dim=-1, keepdim=True)).rsqrt_()
        c_factor = exp_avg_sq_col.rsqrt()
        return torch.mul(r_factor.unsqueeze(-1), c_factor.unsqueeze(-2))

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad
                if grad.dtype in {torch.float16, torch.bfloat16}:
                    grad = grad.float()
                if grad.is_sparse:
                    raise RuntimeError("Adafactor does not support sparse gradients.")

                state = self.state[p]
                grad_shape = grad.shape

                factored = grad.dim() >= 2
                use_first_moment = group["beta1"] is not None

                # ── Initialise state ──────────────────────────────────────────
                if len(state) == 0:
                    state["step"] = 0
                    if use_first_moment:
                        state["exp_avg"] = torch.zeros_like(p)
                    if factored:
                        state["exp_avg_sq_row"] = torch.zeros(grad_shape[:-1]).to(dtype=torch.float, device=p.device)
                        state["exp_avg_sq_col"] = torch.zeros(grad_shape[:-2] + grad_shape[-1:]).to(dtype=torch.float, device=p.device)
                    else:
                        state["exp_avg_sq"] = torch.zeros_like(p)
                    state["rms"] = 0.0

                state["step"] += 1
                state["rms"]   = self._rms(p.data.float())
                lr             = self._get_lr(group, state)

                beta2t = 1.0 - math.pow(state["step"], group["decay_rate"])
                update = (grad ** 2) + group["eps"][0]

                # ── Second moment update ───────────────────────────────────────
                if factored:
                    row = state["exp_avg_sq_row"].float()
                    col = state["exp_avg_sq_col"].float()
                    row.mul_(beta2t).add_(update.mean(dim=-1),  alpha=1.0 - beta2t)
                    col.mul_(beta2t).add_(update.mean(dim=-2),  alpha=1.0 - beta2t)
                    state["exp_avg_sq_row"] = row
                    state["exp_avg_sq_col"] = col
                    update = self._approx_sq_grad(row, col)
                    update.mul_(grad)
                else:
                    v = state["exp_avg_sq"].float()
                    v.mul_(beta2t).add_(update, alpha=1.0 - beta2t)
                    state["exp_avg_sq"] = v
                    update = v.rsqrt().mul_(grad)

                # ── Clip update norm ──────────────────────────────────────────
                update_rms = self._rms(update)
                if update_rms > group["clip_threshold"]:
                    update.mul_(group["clip_threshold"] / update_rms)

                update.mul_(lr)

                # ── First moment ──────────────────────────────────────────────
                if use_first_moment:
                    exp_avg = state["exp_avg"].float()
                    exp_avg.mul_(group["beta1"]).add_(update, alpha=1.0 - group["beta1"])
                    update = exp_avg
                    state["exp_avg"] = exp_avg

                # ── Weight decay ──────────────────────────────────────────────
                if group["weight_decay"] != 0:
                    p.data.add_(p.data, alpha=-group["weight_decay"] * lr)

                p.data.add_(-update.to(p.data.dtype))

        return loss


def build_optimizer(model: torch.nn.Module, cfg) -> Adafactor:
    """Construct Adafactor from TrainConfig."""
    return Adafactor(
        model.parameters(),
        lr              = cfg.learning_rate,
        scale_parameter = cfg.adafactor_scale_parameter,
        relative_step   = cfg.adafactor_relative_step,
        warmup_init     = False,
        clip_threshold  = cfg.clip_grad_norm,
        weight_decay    = cfg.weight_decay,
    )
