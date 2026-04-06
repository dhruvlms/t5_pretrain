"""
config.py
─────────
Single source of truth for all hyperparameters and paths.

Automatically detects the runtime environment and sets paths so that
checkpoints and data are ALWAYS saved to permanent storage:

  Colab   → Google Drive  (/drive/MyDrive/t5_pretrain/)
  Kaggle  → Kaggle output (/kaggle/working/t5_pretrain/)
  Local   → project root  (./data_store/, ./checkpoints/)

No manual path changes needed — push to GitHub once, works everywhere.
"""

import os
from dataclasses import dataclass


# ─────────────────────────────────────────────────────────────────────────────
#  Environment detection
# ─────────────────────────────────────────────────────────────────────────────

def _detect_env() -> str:
    """Returns 'colab', 'kaggle', or 'local'."""
    try:
        import google.colab  # noqa: F401
        return "colab"
    except ImportError:
        pass
    if os.path.exists("/kaggle/working"):
        return "kaggle"
    return "local"


ENV = _detect_env()


# ─────────────────────────────────────────────────────────────────────────────
#  Persistent storage root — changes per environment
# ─────────────────────────────────────────────────────────────────────────────

def _get_storage_root() -> str:
    """
    Returns the root directory for ALL persistent data
    (data, checkpoints, logs).

    Colab  → /drive/MyDrive/t5_pretrain
             (Google Drive — survives session disconnection)

    Kaggle → /kaggle/working/t5_pretrain
             (Kaggle output — persists between sessions)

    Local  → ./  (project root — always persistent on your disk)
    """
    if ENV == "colab":
        drive_root = "/drive/MyDrive/t5_pretrain"
        # Warn clearly if Drive is not mounted yet
        if not os.path.exists("/drive/MyDrive"):
            print("=" * 60)
            print("WARNING: Google Drive is NOT mounted!")
            print("Run this cell BEFORE importing config:")
            print()
            print("    from google.colab import drive")
            print("    drive.mount('/drive')")
            print()
            print("Checkpoints will be lost when session ends if")
            print("Drive is not mounted.")
            print("=" * 60)
            # Fall back to /content so training can still start
            # but warn loudly
            return "/content/t5_pretrain_TEMP_NO_DRIVE"
        return drive_root

    elif ENV == "kaggle":
        return "/kaggle/working/t5_pretrain"

    else:  # local
        return os.path.dirname(os.path.abspath(__file__))


# ─────────────────────────────────────────────────────────────────────────────
#  Paths
# ─────────────────────────────────────────────────────────────────────────────

_STORAGE = _get_storage_root()
_CODE    = os.path.dirname(os.path.abspath(__file__))  # always the repo root

# Data — large files, must be persistent
DATA_DIR        = os.path.join(_STORAGE, "data_store")
RAW_TEXT_FILE   = os.path.join(DATA_DIR, "c4_raw.txt")
TOKENIZER_DIR   = os.path.join(DATA_DIR, "tokenizer")
TOKENIZED_DIR   = os.path.join(DATA_DIR, "tokenized")

# Outputs — must be persistent (survive session restart)
CHECKPOINT_DIR  = os.path.join(_STORAGE, "checkpoints")
LOG_DIR         = os.path.join(_STORAGE, "logs")

# Code — always the cloned repo (temporary is fine, re-clone each session)
BASE_DIR        = _CODE

# Create all directories immediately so nothing fails later
for _d in [DATA_DIR, TOKENIZER_DIR, TOKENIZED_DIR, CHECKPOINT_DIR, LOG_DIR]:
    os.makedirs(_d, exist_ok=True)

# Print path summary so user can verify on startup
print(f"Environment  : {ENV.upper()}")
print(f"Storage root : {_STORAGE}")
print(f"Checkpoints  : {CHECKPOINT_DIR}")
print(f"Logs         : {LOG_DIR}")
print(f"Data         : {DATA_DIR}")
print()


# ─────────────────────────────────────────────────────────────────────────────
#  Tokenizer
# ─────────────────────────────────────────────────────────────────────────────

VOCAB_SIZE          = 32_128
NUM_SENTINEL_TOKENS = 100
PAD_ID              = 0
EOS_ID              = 1
UNK_ID              = 2


# ─────────────────────────────────────────────────────────────────────────────
#  Data
# ─────────────────────────────────────────────────────────────────────────────

TARGET_TOKENS    = 300_000_000
TRAIN_FRAC       = 0.90
VAL_FRAC         = 0.05
TEST_FRAC        = 0.05
MAX_SEQ_LEN      = 512
MAX_TARGET_LEN   = 114
SHARD_SIZE       = 10_000_000


# ─────────────────────────────────────────────────────────────────────────────
#  Span corruption
# ─────────────────────────────────────────────────────────────────────────────

NOISE_DENSITY    = 0.15
MEAN_SPAN_LENGTH = 3


# ─────────────────────────────────────────────────────────────────────────────
#  Model — ~60M parameters
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ModelConfig:
    vocab_size:                 int   = VOCAB_SIZE
    d_model:                    int   = 512
    d_ff:                       int   = 2048
    num_heads:                  int   = 8
    d_kv:                       int   = 64
    num_encoder_layers:         int   = 6
    num_decoder_layers:         int   = 6
    dropout:                    float = 0.1
    max_seq_len:                int   = MAX_SEQ_LEN
    pad_id:                     int   = PAD_ID
    eos_id:                     int   = EOS_ID
    relative_attn_buckets:      int   = 32
    relative_attn_max_distance: int   = 128


# ─────────────────────────────────────────────────────────────────────────────
#  Training
#
#  max_steps is set to 9,175 = 4 epochs (Chinchilla optimal for 60M params)
#  Token budget: 20 × 60M = 1.2B tokens
#  tokens_per_step = 256 × 512 = 131,072
#  steps = 1.2B / 131,072 = 9,175
# ─────────────────────────────────────────────────────────────────────────────

def _batch_for_env() -> tuple:
    """
    Returns (batch_size, grad_accum_steps) tuned for each environment.
    Effective batch size is always 256 regardless of GPU.
    """
    if ENV == "colab" or ENV == "kaggle":
        # T4 = 16GB VRAM
        return 32, 8     # 32 × 8 = 256 ✅
    else:
        # Local — assume A100 or similar, or CPU debug
        if not _cuda_available():
            return 4, 1  # CPU debug — tiny batch
        return 128, 2    # A100 — 128 × 2 = 256 ✅


def _cuda_available() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


_batch, _accum = _batch_for_env()


@dataclass
class TrainConfig:
    # ── Batch ────────────────────────────────────────────────────────────────
    batch_size:          int   = _batch   # auto-set per environment
    grad_accum_steps:    int   = _accum   # effective batch always = 256

    # ── Steps (Chinchilla optimal: 4 epochs on 300M tokens for 60M params) ──
    max_steps:           int   = 9_175    # 1.2B tokens total
    warmup_steps:        int   = 500      # ~5% of max_steps
    eval_every:          int   = 200      # validate every 200 steps
    save_every:          int   = 200      # checkpoint every 200 steps
    log_every:           int   = 20       # log every 20 steps

    # ── Optimiser (Adafactor) ─────────────────────────────────────────────────
    learning_rate:               float = 1e-2
    weight_decay:                float = 0.0
    clip_grad_norm:              float = 1.0
    adafactor_scale_parameter:   bool  = True
    adafactor_relative_step:     bool  = False

    # ── Early stopping ────────────────────────────────────────────────────────
    patience:            int   = 5
    min_delta:           float = 1e-4

    # ── Misc ──────────────────────────────────────────────────────────────────
    seed:                int   = 42
    num_workers:         int   = 2 if ENV in ("colab", "kaggle") else 4
    pin_memory:          bool  = True
    fp16:                bool  = True
    compile_model:       bool  = False

    # ── Logging ───────────────────────────────────────────────────────────────
    use_wandb:           bool  = False
    wandb_project:       str   = "t5-60m-pretrain"
    run_name:            str   = "span-corruption-c4-300M"


MODEL_CFG = ModelConfig()
TRAIN_CFG = TrainConfig()
