"""
train.py  —  ENTRY POINT
────────────────────────
Run this to start (or resume) pretraining:

    # Fresh run:
    python train.py

    # Resume from a checkpoint:
    python train.py --resume checkpoints/latest.pt

    # Override hyperparameters via CLI:
    python train.py --batch_size 64 --fp16 false

Prerequisites (run once before training):
    python data/download_c4.py
    python data/tokenizer_train.py
    python data/preprocess.py
"""

import argparse
import logging
import os
import random

import numpy as np
import torch

from config import MODEL_CFG, TRAIN_CFG, CHECKPOINT_DIR
from data.dataset import make_dataloaders
from model.transformer import T5Model, count_parameters
from training.trainer import Trainer

# ─────────────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level   = logging.INFO,
    format  = "%(asctime)s  [%(levelname)s]  %(message)s",
    handlers= [
        logging.StreamHandler(),
        logging.FileHandler(os.path.join("logs", "train.log"), mode="a"),
    ],
)
log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_args():
    p = argparse.ArgumentParser(description="T5-style pretraining on C4")
    p.add_argument("--resume",       type=str,   default=None,  help="Path to checkpoint to resume from")
    p.add_argument("--batch_size",   type=int,   default=None,  help="Override batch size")
    p.add_argument("--max_steps",    type=int,   default=None,  help="Override max training steps")
    p.add_argument("--lr",           type=float, default=None,  help="Override peak learning rate")
    p.add_argument("--fp16",         type=str,   default=None,  help="Enable mixed precision: true/false")
    p.add_argument("--compile",      action="store_true",       help="Enable torch.compile (PyTorch 2.0+)")
    p.add_argument("--no_wandb",     action="store_true",       help="Disable WandB logging")
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
def main():
    os.makedirs("logs", exist_ok=True)
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    args = parse_args()

    # ── Apply CLI overrides ──────────────────────────────────────────────────
    if args.batch_size is not None: TRAIN_CFG.batch_size   = args.batch_size
    if args.max_steps  is not None: TRAIN_CFG.max_steps    = args.max_steps
    if args.lr         is not None: TRAIN_CFG.learning_rate= args.lr
    if args.fp16       is not None: TRAIN_CFG.fp16         = args.fp16.lower() == "true"
    if args.compile:                TRAIN_CFG.compile_model = True
    if args.no_wandb:               TRAIN_CFG.use_wandb    = False

    set_seed(TRAIN_CFG.seed)

    # ── Device info ──────────────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("Device: %s", device)
    if device.type == "cuda":
        log.info("  GPU: %s  |  VRAM: %.1f GB",
                 torch.cuda.get_device_name(0),
                 torch.cuda.get_device_properties(0).total_memory / 1e9)

    # ── Model ────────────────────────────────────────────────────────────────
    log.info("Building model …")
    model  = T5Model(MODEL_CFG)
    n_params = count_parameters(model)
    log.info("Parameters: %d  (%.1f M)", n_params, n_params / 1e6)

    # Log model config
    log.info("Model config: d_model=%d  d_ff=%d  heads=%d  enc_layers=%d  dec_layers=%d",
             MODEL_CFG.d_model, MODEL_CFG.d_ff, MODEL_CFG.num_heads,
             MODEL_CFG.num_encoder_layers, MODEL_CFG.num_decoder_layers)
    log.info("Train config: batch=%d  grad_accum=%d  effective_batch=%d  max_steps=%d",
             TRAIN_CFG.batch_size, TRAIN_CFG.grad_accum_steps,
             TRAIN_CFG.batch_size * TRAIN_CFG.grad_accum_steps,
             TRAIN_CFG.max_steps)

    # ── Data ─────────────────────────────────────────────────────────────────
    log.info("Building dataloaders …")
    train_dl, val_dl, test_dl = make_dataloaders(
        batch_size  = TRAIN_CFG.batch_size,
        num_workers = TRAIN_CFG.num_workers,
        pin_memory  = TRAIN_CFG.pin_memory and device.type == "cuda",
        seed        = TRAIN_CFG.seed,
    )

    # ── Trainer ──────────────────────────────────────────────────────────────
    trainer = Trainer(
        model       = model,
        train_dl    = train_dl,
        val_dl      = val_dl,
        test_dl     = test_dl,
        cfg         = TRAIN_CFG,
        device      = device,
        resume_from = args.resume,
    )

    # ── Train ────────────────────────────────────────────────────────────────
    log.info("=" * 60)
    log.info("Starting pretraining …")
    log.info("=" * 60)
    test_loss = trainer.train()

    log.info("=" * 60)
    log.info("Pretraining finished.  Final test loss: %.4f", test_loss)
    log.info("Best checkpoint: %s/best_model.pt", CHECKPOINT_DIR)


if __name__ == "__main__":
    main()
