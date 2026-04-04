"""
data/dataset.py
───────────────
PyTorch Dataset that:
  1. Memory-maps pre-tokenized NumPy shards from disk.
  2. Samples fixed-length sequences (MAX_SEQ_LEN tokens).
  3. Applies DYNAMIC span corruption on-the-fly each epoch (T5 style):
       - Randomly selects ~15 % of tokens using a geometric span distribution.
       - Replaces each span with a single <extra_id_X> sentinel.
       - Returns:
           encoder_input  (corrupted sequence)
           decoder_target (sentinel + original span + </s>)
  4. Pads / truncates to fixed lengths so batches can be collated.

Usage:
    from data.dataset import make_dataloaders
    train_dl, val_dl, test_dl = make_dataloaders()
"""

import os, sys, glob, random
from typing import Tuple, List, Optional

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    TOKENIZED_DIR, VOCAB_SIZE, NUM_SENTINEL_TOKENS,
    MAX_SEQ_LEN, MAX_TARGET_LEN, TRAIN_CFG,
    PAD_ID, EOS_ID,
    NOISE_DENSITY, MEAN_SPAN_LENGTH,
)


# ─────────────────────────────────────────────────────────────────────────────
#  Sentinel helper
# ─────────────────────────────────────────────────────────────────────────────
def sentinel_id(idx: int) -> int:
    """<extra_id_idx> token id  (T5 convention: counts down from top of vocab)."""
    return VOCAB_SIZE - 1 - idx          # <extra_id_0>=32127, <extra_id_1>=32126 …


# ─────────────────────────────────────────────────────────────────────────────
#  Span-corruption logic
# ─────────────────────────────────────────────────────────────────────────────
def _sample_span_mask(length: int,
                      noise_density: float = NOISE_DENSITY,
                      mean_span_length: float = MEAN_SPAN_LENGTH) -> np.ndarray:
    """
    Returns a boolean mask of shape (length,) where True = token is masked.
    Uses a geometric distribution for span lengths (T5 Appendix A).
    """
    # geometric span length p
    p = 1.0 / mean_span_length
    num_noise_tokens = max(1, int(round(length * noise_density)))

    mask = np.zeros(length, dtype=bool)
    num_masked = 0
    max_tries  = length * 10

    tries = 0
    while num_masked < num_noise_tokens and tries < max_tries:
        span_len = max(1, int(np.random.geometric(p)))
        start    = random.randint(0, length - 1)
        end      = min(start + span_len, length)
        actual   = end - start
        if num_masked + actual > num_noise_tokens * 2:   # don't overshoot badly
            tries += 1
            continue
        mask[start:end] = True
        num_masked += actual
        tries += 1

    return mask


def apply_span_corruption(
    tokens: np.ndarray,
    noise_density: float = NOISE_DENSITY,
    mean_span_length: float = MEAN_SPAN_LENGTH,
    max_sentinels: int = NUM_SENTINEL_TOKENS,
) -> Tuple[List[int], List[int]]:
    """
    Given a 1-D token array, returns:
        encoder_input  : list[int]  (corrupted, with <extra_id_X> replacing spans)
        decoder_target : list[int]  (<extra_id_X> + original span tokens … </s>)
    """
    mask = _sample_span_mask(len(tokens), noise_density, mean_span_length)

    encoder_input:  List[int] = []
    decoder_target: List[int] = []
    sentinel_idx = 0
    in_span = False

    for i, tok in enumerate(tokens.tolist()):
        if mask[i]:
            if not in_span:
                # start of a new masked span
                if sentinel_idx >= max_sentinels:
                    # ran out of sentinels — treat remainder as unmasked
                    encoder_input.append(tok)
                    in_span = False
                    continue
                sid = sentinel_id(sentinel_idx)
                encoder_input.append(sid)
                decoder_target.append(sid)
                sentinel_idx += 1
                in_span = True
            decoder_target.append(tok)
        else:
            in_span = False
            encoder_input.append(tok)

    decoder_target.append(EOS_ID)   # </s> at end of target
    return encoder_input, decoder_target


def pad_or_truncate(seq: List[int], max_len: int, pad_id: int = PAD_ID) -> Tuple[List[int], List[int]]:
    """Returns (padded_ids, attention_mask) both of length max_len."""
    seq = seq[:max_len]
    mask = [1] * len(seq) + [0] * (max_len - len(seq))
    seq  = seq              + [pad_id] * (max_len - len(seq))
    return seq, mask


# ─────────────────────────────────────────────────────────────────────────────
#  Dataset
# ─────────────────────────────────────────────────────────────────────────────
class SpanCorruptionDataset(Dataset):
    """
    Streams from memory-mapped NumPy shards.
    Each call to __getitem__ draws a contiguous MAX_SEQ_LEN slice from the
    flat token stream and applies span corruption dynamically.
    """

    def __init__(self, split: str, seq_len: int = MAX_SEQ_LEN, seed: Optional[int] = None):
        assert split in ("train", "val", "test")
        self.seq_len = seq_len
        self.split   = split

        shard_paths = sorted(glob.glob(os.path.join(TOKENIZED_DIR, split, "shard_*.npy")))
        if not shard_paths:
            raise FileNotFoundError(
                f"No shards found in {TOKENIZED_DIR}/{split}/\n"
                "Run:  python data/preprocess.py"
            )

        # Memory-map all shards and concatenate virtually via offset table
        self._shards  = [np.load(p, mmap_mode="r") for p in shard_paths]
        self._offsets = np.cumsum([0] + [len(s) for s in self._shards])
        total_tokens  = int(self._offsets[-1])

        # Number of non-overlapping samples
        self._n_samples = (total_tokens - seq_len) // seq_len

        self.rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return self._n_samples

    def _get_tokens(self, flat_idx: int) -> np.ndarray:
        """Fetch seq_len tokens starting at flat_idx across shards."""
        shard_idx = int(np.searchsorted(self._offsets, flat_idx, side="right")) - 1
        local_off  = flat_idx - int(self._offsets[shard_idx])
        shard = self._shards[shard_idx]

        end_local = local_off + self.seq_len
        if end_local <= len(shard):
            return np.array(shard[local_off:end_local], dtype=np.int32)

        # spans two shards
        part1 = np.array(shard[local_off:], dtype=np.int32)
        need  = self.seq_len - len(part1)
        next_shard = self._shards[shard_idx + 1]
        part2 = np.array(next_shard[:need], dtype=np.int32)
        return np.concatenate([part1, part2])

    def __getitem__(self, idx: int) -> dict:
        flat_idx = idx * self.seq_len
        tokens   = self._get_tokens(flat_idx)

        enc_ids, dec_ids = apply_span_corruption(tokens)

        enc_ids, enc_mask = pad_or_truncate(enc_ids, MAX_SEQ_LEN)
        dec_ids, dec_mask = pad_or_truncate(dec_ids, MAX_TARGET_LEN)

        # decoder labels = dec_ids shifted: ignore PAD positions with -100
        labels = [t if m == 1 else -100 for t, m in zip(dec_ids, dec_mask)]

        return {
            "input_ids":              torch.tensor(enc_ids,  dtype=torch.long),
            "attention_mask":         torch.tensor(enc_mask, dtype=torch.long),
            "decoder_input_ids":      torch.tensor(dec_ids,  dtype=torch.long),
            "decoder_attention_mask": torch.tensor(dec_mask, dtype=torch.long),
            "labels":                 torch.tensor(labels,   dtype=torch.long),
        }


# ─────────────────────────────────────────────────────────────────────────────
#  DataLoader factory
# ─────────────────────────────────────────────────────────────────────────────
def make_dataloaders(
    batch_size:  int  = TRAIN_CFG.batch_size,
    num_workers: int  = TRAIN_CFG.num_workers,
    pin_memory:  bool = TRAIN_CFG.pin_memory,
    seed:        int  = TRAIN_CFG.seed,
) -> Tuple[DataLoader, DataLoader, DataLoader]:

    train_ds = SpanCorruptionDataset("train", seed=seed)
    val_ds   = SpanCorruptionDataset("val",   seed=seed + 1)
    test_ds  = SpanCorruptionDataset("test",  seed=seed + 2)

    g = torch.Generator()
    g.manual_seed(seed)

    train_dl = DataLoader(
        train_ds,
        batch_size  = batch_size,
        shuffle     = True,
        num_workers = num_workers,
        pin_memory  = pin_memory,
        generator   = g,
        drop_last   = True,
    )
    val_dl = DataLoader(
        val_ds,
        batch_size  = batch_size * 2,
        shuffle     = False,
        num_workers = num_workers,
        pin_memory  = pin_memory,
        drop_last   = False,
    )
    test_dl = DataLoader(
        test_ds,
        batch_size  = batch_size * 2,
        shuffle     = False,
        num_workers = num_workers,
        pin_memory  = pin_memory,
        drop_last   = False,
    )

    print(f"Dataset sizes  |  train={len(train_ds):,}  val={len(val_ds):,}  test={len(test_ds):,}")
    return train_dl, val_dl, test_dl
