"""
data/preprocess.py
──────────────────
Tokenizes c4_raw.txt with the trained SentencePiece tokenizer, appends
</s> (EOS) after every document, and writes the flat token stream to
numbered NumPy shards under data_store/tokenized/.

Split is done by token count (not document) after all tokens are written:
    train : 90 %
    val   :  5 %
    test  :  5 %

Run once (after tokenizer_train.py):
    python data/preprocess.py

Output:
    data_store/tokenized/train/shard_0000.npy  …
    data_store/tokenized/val/shard_0000.npy    …
    data_store/tokenized/test/shard_0000.npy   …
    data_store/tokenized/meta.json             — split sizes in tokens
"""

import os, sys, json, logging
from pathlib import Path
from typing import List

import numpy as np
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    DATA_DIR, RAW_TEXT_FILE, TOKENIZER_DIR, TOKENIZED_DIR,
    SHARD_SIZE, EOS_ID, TRAIN_FRAC, VAL_FRAC
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s")
log = logging.getLogger(__name__)

try:
    import sentencepiece as spm
except ImportError:
    raise SystemExit("Run:  pip install sentencepiece")


# ─────────────────────────────────────────────────────────────────────────────
def count_lines(path: str) -> int:
    n = 0
    with open(path, "rb") as f:
        for _ in f:
            n += 1
    return n


def tokenize_corpus() -> np.ndarray:
    """Tokenize entire corpus into one large uint16 array."""
    model_path = os.path.join(TOKENIZER_DIR, "tokenizer.model")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Tokenizer not found: {model_path}\nRun tokenizer_train.py first.")

    sp = spm.SentencePieceProcessor(model_file=model_path)
    log.info("Tokenizer loaded  (vocab=%d)", sp.get_piece_size())

    n_lines = count_lines(RAW_TEXT_FILE)
    log.info("Tokenizing %d documents …", n_lines)

    all_ids: List[np.ndarray] = []
    total_tokens = 0

    with open(RAW_TEXT_FILE, "r", encoding="utf-8") as f:
        for line in tqdm(f, total=n_lines, unit="doc"):
            line = line.strip()
            if not line:
                continue
            ids = sp.encode(line, out_type=int)
            ids.append(EOS_ID)                        # </s> after every doc
            all_ids.append(np.array(ids, dtype=np.uint16))
            total_tokens += len(ids)

    log.info("Total tokens (with EOS): %d", total_tokens)
    return np.concatenate(all_ids)


def write_shards(tokens: np.ndarray, split_dir: str, shard_size: int) -> int:
    os.makedirs(split_dir, exist_ok=True)
    n = len(tokens)
    n_shards = (n + shard_size - 1) // shard_size
    for i in range(n_shards):
        shard = tokens[i * shard_size : (i + 1) * shard_size]
        path  = os.path.join(split_dir, f"shard_{i:04d}.npy")
        np.save(path, shard)
    log.info("  %s  →  %d shards  (%d tokens)", split_dir, n_shards, n)
    return n


def preprocess() -> None:
    meta_path = os.path.join(TOKENIZED_DIR, "meta.json")
    if os.path.exists(meta_path):
        log.info("Tokenized data already exists (%s). Delete to re-process.", meta_path)
        return

    os.makedirs(TOKENIZED_DIR, exist_ok=True)

    # ── 1. Tokenize ──────────────────────────────────────────────────────────
    all_tokens = tokenize_corpus()
    N = len(all_tokens)

    # ── 2. Split ─────────────────────────────────────────────────────────────
    train_end = int(N * TRAIN_FRAC)
    val_end   = int(N * (TRAIN_FRAC + VAL_FRAC))

    train_tokens = all_tokens[:train_end]
    val_tokens   = all_tokens[train_end:val_end]
    test_tokens  = all_tokens[val_end:]

    log.info("Split  |  train=%d  val=%d  test=%d  (tokens)", len(train_tokens), len(val_tokens), len(test_tokens))

    # ── 3. Write shards ──────────────────────────────────────────────────────
    n_train = write_shards(train_tokens, os.path.join(TOKENIZED_DIR, "train"), SHARD_SIZE)
    n_val   = write_shards(val_tokens,   os.path.join(TOKENIZED_DIR, "val"),   SHARD_SIZE)
    n_test  = write_shards(test_tokens,  os.path.join(TOKENIZED_DIR, "test"),  SHARD_SIZE)

    # ── 4. Metadata ──────────────────────────────────────────────────────────
    meta = {"total": N, "train": n_train, "val": n_val, "test": n_test}
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    log.info("Metadata → %s", meta_path)
    log.info("Preprocessing complete.")


if __name__ == "__main__":
    preprocess()
