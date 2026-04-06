"""
training/scheduler.py
──────────────────────
Inverse-square-root LR schedule with linear warmup — the standard schedule
used to train T5 when supplying an explicit LR to Adafactor.

    lr(t) = peak_lr × min(1, t / warmup_steps) / max(1, sqrt(t / warmup_steps))

which simplifies to:
    t ≤ warmup  →  lr = peak_lr × t / warmup_steps   (linear ramp)
    t >  warmup →  lr = peak_lr × sqrt(warmup_steps / t)  (decay)
"""

import math
from torch.optim.lr_scheduler import LambdaLR
from torch.optim import Optimizer


def get_inverse_sqrt_schedule(
    optimizer:     Optimizer,
    warmup_steps:  int,
    last_epoch:    int = -1,
) -> LambdaLR:
    """
    Returns a scheduler whose get_lr() multiplier is:
        min(1, step/warmup) * sqrt(warmup / max(step, warmup))
    """
    def lr_lambda(current_step: int) -> float:
        current_step = max(1, current_step)
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        return math.sqrt(warmup_steps / current_step)

    return LambdaLR(optimizer, lr_lambda, last_epoch=last_epoch)
