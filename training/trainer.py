"""
training/trainer.py
───────────────────
Main Trainer class.  Owns the training loop, validation loop, mixed-precision
scaler, gradient accumulation, and callback dispatch.

Usage (via train.py):
    trainer = Trainer(model, train_dl, val_dl, test_dl)
    trainer.train()
"""

import os, sys, time, logging, math
from typing import Optional

import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import TRAIN_CFG, CHECKPOINT_DIR
from training.optimizer  import build_optimizer
from training.scheduler  import get_inverse_sqrt_schedule
from training.callbacks  import (
    CheckpointCallback, EarlyStoppingCallback,
    LoggingCallback, MetricsCallback,
)

log = logging.getLogger(__name__)


class Trainer:
    def __init__(
        self,
        model:    nn.Module,
        train_dl: DataLoader,
        val_dl:   DataLoader,
        test_dl:  DataLoader,
        cfg       = TRAIN_CFG,
        device:   Optional[torch.device] = None,
        resume_from: Optional[str] = None,
    ):
        self.cfg      = cfg
        self.model    = model
        self.train_dl = train_dl
        self.val_dl   = val_dl
        self.test_dl  = test_dl

        # Device
        self.device = device or (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )
        self.model.to(self.device)

        # Optional torch.compile (PyTorch 2.0+, ~20-30 % faster on A100)
        if cfg.compile_model and hasattr(torch, "compile"):
            log.info("Compiling model with torch.compile …")
            self.model = torch.compile(self.model)

        # Optimizer + scheduler
        self.optimizer = build_optimizer(model, cfg)
        self.scheduler = get_inverse_sqrt_schedule(self.optimizer, cfg.warmup_steps)

        # Mixed precision
        self.scaler    = GradScaler(device="cuda", enabled=(cfg.fp16 and self.device.type == "cuda"))

        # State
        self.global_step = 0
        self.should_stop = False

        # Callbacks
        self.ckpt_cb    = CheckpointCallback()
        self.es_cb      = EarlyStoppingCallback()
        self.log_cb     = LoggingCallback(cfg)
        self.metrics_cb = MetricsCallback()

        # Resume
        if resume_from and os.path.exists(resume_from):
            ckpt = CheckpointCallback.load(
                resume_from, model, self.optimizer, self.scheduler, device=self.device
            )
            self.global_step = ckpt.get("step", 0)
            log.info("Resumed from step %d", self.global_step)

        log.info("Trainer ready  |  device=%s  fp16=%s  grad_accum=%d",
                 self.device, cfg.fp16, cfg.grad_accum_steps)

    # ── validation loop ───────────────────────────────────────────────────────
    @torch.no_grad()
    def _evaluate(self, loader: DataLoader, max_batches: int = 200) -> float:
        """
        Compute average loss over `max_batches` batches.
        We cap at 200 batches so validation is fast (~seconds) during training.
        Pass max_batches=None for a full eval (used on test set).
        """
        self.model.eval()
        total_loss = 0.0
        n_batches  = 0

        for batch in loader:
            batch = {k: v.to(self.device, non_blocking=True) for k, v in batch.items()}
            with autocast(device_type="cuda", enabled=(self.cfg.fp16 and self.device.type == "cuda")):
                out = self.model(**batch)
            total_loss += out["loss"].item()
            n_batches  += 1
            if max_batches and n_batches >= max_batches:
                break

        self.model.train()
        return total_loss / max(1, n_batches)

    # ── main train loop ───────────────────────────────────────────────────────
    def train(self) -> None:
        cfg   = self.cfg
        model = self.model
        model.train()

        log.info("Starting training  |  max_steps=%d  warmup=%d", cfg.max_steps, cfg.warmup_steps)

        running_loss  = 0.0
        accum_counter = 0
        self.optimizer.zero_grad()

        # Infinite iterator over train dataloader
        step_in_epoch = 0
        train_iter    = iter(self.train_dl)

        while self.global_step < cfg.max_steps and not self.should_stop:
            # ── Get batch (restart iterator at epoch boundary) ────────────────
            try:
                batch = next(train_iter)
            except StopIteration:
                train_iter    = iter(self.train_dl)
                step_in_epoch = 0
                batch = next(train_iter)

            batch         = {k: v.to(self.device, non_blocking=True) for k, v in batch.items()}
            step_in_epoch += 1

            # ── Forward pass ──────────────────────────────────────────────────
            with autocast(device_type="cuda", enabled=(cfg.fp16 and self.device.type == "cuda")):
                out  = model(**batch)
                loss = out["loss"] / cfg.grad_accum_steps   # scale for accumulation

            # ── Backward ─────────────────────────────────────────────────────
            self.scaler.scale(loss).backward()
            running_loss  += loss.item() * cfg.grad_accum_steps
            accum_counter += 1

            # ── Optimizer step every grad_accum_steps mini-batches ────────────
            if accum_counter % cfg.grad_accum_steps == 0:
                # Unscale before gradient clipping
                self.scaler.unscale_(self.optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), cfg.clip_grad_norm)

                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.scheduler.step()
                self.optimizer.zero_grad()

                self.global_step += 1
                step_loss = running_loss / cfg.grad_accum_steps
                running_loss = 0.0

                current_lr = self.scheduler.get_last_lr()[0]

                # ── Callbacks: train step ─────────────────────────────────────
                self.log_cb.on_train_step(self, step_loss, current_lr)
                self.metrics_cb.on_train_step(self, step_loss, current_lr)
                self.ckpt_cb.on_step(self)

                # ── Validation ────────────────────────────────────────────────
                if self.global_step % cfg.eval_every == 0:
                    val_loss = self._evaluate(self.val_dl, max_batches=200)
                    self.log_cb.on_validation(self, val_loss)
                    self.metrics_cb.on_validation(self, val_loss)
                    self.ckpt_cb.on_validation(self, val_loss)
                    self.es_cb.on_validation(self, val_loss)

                if self.global_step >= cfg.max_steps:
                    break

        # ── Final test evaluation ────────────────────────────────────────────
        log.info("Training complete.  Running test evaluation …")
        test_loss = self._evaluate(self.test_dl, max_batches=None)
        self.log_cb.on_test(self, test_loss)
        self.log_cb.close()

        log.info("Done.  Best checkpoint → %s/best_model.pt", CHECKPOINT_DIR)
        return test_loss
