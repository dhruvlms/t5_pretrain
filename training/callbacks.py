"""
training/callbacks.py  (FIXED)
──────────────────────────────
Fixes applied vs original:

  Bug 1 — latest.pt was saved EVERY step (extremely slow, wasteful).
           Fixed: latest.pt now saves every `save_every` steps only.

  Bug 2 — No atomic write protection. If Colab dies mid-save, the .pt
           file gets corrupted and resume fails with a zip error.
           Fixed: all saves write to a .tmp file first, then rename
           atomically. A half-written .tmp never overwrites a good .pt.

  Bug 3 — No verification that the save actually succeeded.
           Fixed: every save is verified by reloading the step field.
           If verification fails, the old file is kept intact.

Callbacks provided:
  CheckpointCallback    — saves best model + periodic snapshots
  EarlyStoppingCallback — stops training when val loss plateaus
  LoggingCallback       — TensorBoard + optional WandB
  MetricsCallback       — tracks loss history for plotting
"""

import os
import sys
import json
import logging
import time
import shutil
from typing import Optional

import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import CHECKPOINT_DIR, LOG_DIR, TRAIN_CFG

log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
#  Atomic save helper
# ─────────────────────────────────────────────────────────────────────────────

def _atomic_save(state: dict, path: str, verify: bool = True) -> bool:
    """
    Save `state` to `path` atomically:
      1. Write to   path + ".tmp"
      2. Verify the .tmp is readable and contains the correct step
      3. Rename .tmp -> path  (atomic on Linux/macOS, near-atomic on Windows)

    Returns True if save succeeded, False if it failed (old file kept safe).
    """
    tmp_path = path + ".tmp"

    try:
        torch.save(state, tmp_path)
    except Exception as e:
        log.error("Save FAILED (write error) for %s: %s", path, e)
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        return False

    # Verify the .tmp before replacing the good file
    if verify:
        try:
            check = torch.load(tmp_path, map_location="cpu")
            saved_step = check.get("step", None)
            if saved_step != state["step"]:
                raise ValueError(
                    f"Step mismatch: wrote {state['step']} but read back {saved_step}"
                )
        except Exception as e:
            log.error("Save FAILED (verification error) for %s: %s", path, e)
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
            return False

    # Atomically replace the old file
    try:
        os.replace(tmp_path, path)
    except Exception as e:
        log.error("Save FAILED (rename error) for %s: %s", path, e)
        return False

    return True


# ─────────────────────────────────────────────────────────────────────────────
#  CheckpointCallback
# ─────────────────────────────────────────────────────────────────────────────

class CheckpointCallback:
    """
    Saves checkpoints at three points:

      latest.pt          - every `save_every` steps  (for resuming)
      step_XXXXXXX.pt    - every `save_every` steps  (permanent snapshots)
      best_model.pt      - whenever val loss improves (best model ever)

    All saves are atomic - a Colab crash mid-save will never corrupt
    an existing good checkpoint.
    """

    def __init__(
        self,
        save_dir:   str = CHECKPOINT_DIR,
        save_every: int = TRAIN_CFG.save_every,
    ):
        os.makedirs(save_dir, exist_ok=True)
        self.save_dir   = save_dir
        self.save_every = save_every
        self.best_val   = float("inf")

        log.info(
            "CheckpointCallback initialised  |  dir=%s  save_every=%d",
            save_dir, save_every,
        )

    def _state(self, trainer) -> dict:
        # Handle torch.compile wrapping
        model = trainer.model
        if hasattr(model, "_orig_mod"):
            model = model._orig_mod
        return {
            "step":            trainer.global_step,
            "model_state":     model.state_dict(),
            "optimizer_state": trainer.optimizer.state_dict(),
            "scheduler_state": trainer.scheduler.state_dict(),
            "best_val_loss":   self.best_val,
            "config":          model.cfg.__dict__,
        }

    def on_step(self, trainer) -> None:
        """
        Save latest.pt and a numbered snapshot every `save_every` steps.
        Does NOT save every single step (that was the original bug).
        """
        if trainer.global_step % self.save_every != 0:
            return

        state = self._state(trainer)

        # 1. latest.pt - always overwrite with newest state
        latest_path = os.path.join(self.save_dir, "latest.pt")
        ok = _atomic_save(state, latest_path)
        if ok:
            log.info(
                "Checkpoint saved  step=%d  ->  latest.pt",
                trainer.global_step,
            )
        else:
            log.error(
                "CHECKPOINT SAVE FAILED at step=%d  ->  latest.pt  "
                "(old file kept intact)",
                trainer.global_step,
            )

        # 2. step_XXXXXXX.pt - rolling snapshot (keeps last 3 only)
        snap_path = os.path.join(
            self.save_dir, f"step_{trainer.global_step:07d}.pt"
        )
        ok2 = _atomic_save(state, snap_path)
        if ok2:
            log.info(
                "Snapshot saved    step=%d  ->  %s",
                trainer.global_step, os.path.basename(snap_path),
            )
        else:
            log.error(
                "SNAPSHOT SAVE FAILED at step=%d  ->  %s",
                trainer.global_step, snap_path,
            )

        # 3. Delete old snapshots — keep only 3 most recent
        # Prevents filling Google Drive (45 snapshots × 240MB = 10.8 GB)
        self._cleanup_old_snapshots(keep=3)

    def _cleanup_old_snapshots(self, keep: int = 3) -> None:
        """
        Delete oldest step_XXXXXXX.pt files, keeping only `keep` most recent.
        Never deletes latest.pt or best_model.pt.

        With keep=3 and save_every=200:
          Drive usage from snapshots = 3 × 240MB = 720MB (vs 10.8GB before)
        """
        try:
            snaps = sorted([
                f for f in os.listdir(self.save_dir)
                if f.startswith("step_") and f.endswith(".pt")
            ])  # sorted ascending = oldest first

            to_delete = snaps[:-keep] if len(snaps) > keep else []

            for fname in to_delete:
                path = os.path.join(self.save_dir, fname)
                try:
                    os.remove(path)
                    log.info("Deleted old snapshot: %s", fname)
                except Exception as e:
                    log.warning("Could not delete %s: %s", fname, e)

        except Exception as e:
            log.warning("Snapshot cleanup failed: %s", e)

    def on_validation(self, trainer, val_loss: float) -> None:
        """Save best_model.pt whenever val loss strictly improves."""
        if val_loss < self.best_val:
            self.best_val = val_loss
            path  = os.path.join(self.save_dir, "best_model.pt")
            state = self._state(trainer)
            ok    = _atomic_save(state, path)
            if ok:
                log.info(
                    "New best val loss %.4f  ->  best_model.pt  (step=%d)",
                    val_loss, trainer.global_step,
                )
            else:
                log.error(
                    "BEST MODEL SAVE FAILED at step=%d  val_loss=%.4f",
                    trainer.global_step, val_loss,
                )

    @staticmethod
    def load(
        path:      str,
        model:     nn.Module,
        optimizer  = None,
        scheduler  = None,
        device:    str = "cpu",
    ) -> dict:
        """
        Load checkpoint from `path` into `model`.
        Handles:
          - GPU checkpoint loaded onto CPU (map_location)
          - torch.compile-wrapped models (_orig_mod)
          - Missing optimizer / scheduler states (graceful skip)
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"Checkpoint not found: {path}")

        log.info("Loading checkpoint from %s ...", path)
        ckpt = torch.load(path, map_location=device)

        target_model = model._orig_mod if hasattr(model, "_orig_mod") else model
        target_model.load_state_dict(ckpt["model_state"])

        if optimizer is not None and "optimizer_state" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer_state"])

        if scheduler is not None and "scheduler_state" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler_state"])

        log.info(
            "Checkpoint loaded  step=%d  best_val=%.4f",
            ckpt.get("step", -1),
            ckpt.get("best_val_loss", float("inf")),
        )
        return ckpt

    @staticmethod
    def find_best_checkpoint(save_dir: str = CHECKPOINT_DIR) -> Optional[str]:
        """
        Returns the path of the most recent valid checkpoint, trying:
          1. latest.pt
          2. best_model.pt
          3. Most recent step_XXXXXXX.pt
          4. None if nothing valid found
        """
        candidates = []

        for name in ["latest.pt", "best_model.pt"]:
            p = os.path.join(save_dir, name)
            if os.path.exists(p):
                candidates.append(p)

        if os.path.exists(save_dir):
            snaps = sorted(
                [f for f in os.listdir(save_dir)
                 if f.startswith("step_") and f.endswith(".pt")],
                reverse=True,
            )
            for s in snaps:
                candidates.append(os.path.join(save_dir, s))

        for path in candidates:
            try:
                ckpt = torch.load(path, map_location="cpu")
                step = ckpt.get("step", None)
                if step is not None:
                    log.info("Valid checkpoint found: %s  (step=%d)", path, step)
                    return path
            except Exception as e:
                log.warning("Skipping corrupt checkpoint %s: %s", path, e)

        log.warning("No valid checkpoint found in %s", save_dir)
        return None


# ─────────────────────────────────────────────────────────────────────────────
#  EarlyStoppingCallback
# ─────────────────────────────────────────────────────────────────────────────

class EarlyStoppingCallback:
    """
    Sets trainer.should_stop = True if val loss has not improved
    by at least `min_delta` for `patience` consecutive evaluations.
    """

    def __init__(
        self,
        patience:  int   = TRAIN_CFG.patience,
        min_delta: float = TRAIN_CFG.min_delta,
    ):
        self.patience  = patience
        self.min_delta = min_delta
        self.best      = float("inf")
        self.wait      = 0

    def on_validation(self, trainer, val_loss: float) -> None:
        if val_loss < self.best - self.min_delta:
            self.best = val_loss
            self.wait = 0
        else:
            self.wait += 1
            log.info(
                "EarlyStopping: no improvement  patience %d/%d  "
                "(best=%.4f  current=%.4f)",
                self.wait, self.patience, self.best, val_loss,
            )
            if self.wait >= self.patience:
                log.warning(
                    "Early stopping triggered at step %d. "
                    "Best val loss was %.4f",
                    trainer.global_step, self.best,
                )
                trainer.should_stop = True


# ─────────────────────────────────────────────────────────────────────────────
#  LoggingCallback
# ─────────────────────────────────────────────────────────────────────────────

class LoggingCallback:
    """
    Logs training metrics to:
      - Python logger  (console + train.log file)
      - TensorBoard    (always, if installed)
      - WandB          (optional)
    """

    def __init__(self, cfg=TRAIN_CFG):
        os.makedirs(LOG_DIR, exist_ok=True)
        self.cfg        = cfg
        self._step_time = time.time()

        # TensorBoard
        try:
            from torch.utils.tensorboard import SummaryWriter
            self.tb = SummaryWriter(log_dir=LOG_DIR)
            log.info("TensorBoard logging -> %s", LOG_DIR)
        except Exception:
            self.tb = None
            log.warning("TensorBoard not available. Run: pip install tensorboard")

        # WandB
        self.wb = None
        if cfg.use_wandb:
            try:
                import wandb
                wandb.init(project=cfg.wandb_project, name=cfg.run_name)
                self.wb = wandb
                log.info("WandB logging enabled -> project=%s", cfg.wandb_project)
            except Exception as e:
                log.warning("WandB init failed: %s", e)

    def on_train_step(self, trainer, loss: float, lr: float) -> None:
        if trainer.global_step % self.cfg.log_every != 0:
            return

        now             = time.time()
        elapsed         = now - self._step_time
        self._step_time = now
        secs_per_step   = elapsed / max(1, self.cfg.log_every)
        ppl             = _safe_perplexity(loss)

        log.info(
            "step %7d | loss %.4f | ppl %8.2f | lr %.2e | %.2f s/step",
            trainer.global_step, loss, ppl, lr, secs_per_step,
        )

        if self.tb:
            self.tb.add_scalar("train/loss",       loss,          trainer.global_step)
            self.tb.add_scalar("train/ppl",        ppl,           trainer.global_step)
            self.tb.add_scalar("train/lr",         lr,            trainer.global_step)
            self.tb.add_scalar("train/s_per_step", secs_per_step, trainer.global_step)

        if self.wb:
            self.wb.log(
                {"train/loss": loss, "train/ppl": ppl, "train/lr": lr},
                step=trainer.global_step,
            )

    def on_validation(self, trainer, val_loss: float) -> None:
        ppl = _safe_perplexity(val_loss)
        log.info(
            "-- VALIDATION  step %7d | val_loss %.4f | val_ppl %.2f",
            trainer.global_step, val_loss, ppl,
        )
        if self.tb:
            self.tb.add_scalar("val/loss", val_loss, trainer.global_step)
            self.tb.add_scalar("val/ppl",  ppl,      trainer.global_step)
        if self.wb:
            self.wb.log(
                {"val/loss": val_loss, "val/ppl": ppl},
                step=trainer.global_step,
            )

    def on_test(self, trainer, test_loss: float) -> None:
        ppl = _safe_perplexity(test_loss)
        log.info("== TEST  loss %.4f | ppl %.2f", test_loss, ppl)
        if self.tb:
            self.tb.add_scalar("test/loss", test_loss, 0)
            self.tb.add_scalar("test/ppl",  ppl,       0)
        if self.wb:
            self.wb.log({"test/loss": test_loss, "test/ppl": ppl})

    def close(self) -> None:
        if self.tb:
            self.tb.close()
        if self.wb:
            self.wb.finish()


# ─────────────────────────────────────────────────────────────────────────────
#  MetricsCallback
# ─────────────────────────────────────────────────────────────────────────────

class MetricsCallback:
    """
    Accumulates train/val loss history and writes logs/metrics.json
    after every validation. Use this file to plot loss curves and
    detect overfitting offline.
    """

    def __init__(self):
        self.train_losses: list = []
        self.val_losses:   list = []
        self.steps:        list = []

    def on_train_step(self, trainer, loss: float, lr: float) -> None:
        if trainer.global_step % TRAIN_CFG.log_every == 0:
            self.train_losses.append(round(loss, 6))
            self.steps.append(trainer.global_step)

    def on_validation(self, trainer, val_loss: float) -> None:
        self.val_losses.append({
            "step":     trainer.global_step,
            "val_loss": round(val_loss, 6),
        })
        self._flush()

    def _flush(self) -> None:
        path = os.path.join(LOG_DIR, "metrics.json")
        tmp  = path + ".tmp"
        data = {
            "train_losses": self.train_losses,
            "val_losses":   self.val_losses,
            "steps":        self.steps,
        }
        try:
            with open(tmp, "w") as f:
                json.dump(data, f, indent=2)
            os.replace(tmp, path)   # atomic write
        except Exception as e:
            log.warning("Failed to write metrics.json: %s", e)


# ─────────────────────────────────────────────────────────────────────────────
#  Utility
# ─────────────────────────────────────────────────────────────────────────────

def _safe_perplexity(loss: float) -> float:
    import math
    try:
        return math.exp(min(loss, 20))
    except OverflowError:
        return float("inf")
