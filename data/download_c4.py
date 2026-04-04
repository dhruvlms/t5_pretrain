"""
data/download_c4.py
───────────────────
Downloads the English C4 dataset from HuggingFace Hub (allenai/c4) in
streaming mode and writes plain-text lines to  data_store/c4_raw.txt
until we accumulate TARGET_TOKENS worth of whitespace-split tokens.

Run once:
    python data/download_c4.py

Output:
    data_store/c4_raw.txt   — one document per line, UTF-8
"""

import os
import sys
import logging

# ── allow `import config` from project root ──────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import DATA_DIR, RAW_TEXT_FILE, TARGET_TOKENS

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s")
log = logging.getLogger(__name__)

try:
    from datasets import load_dataset
except ImportError:
    raise SystemExit("Run:  pip install datasets")


def download_c4(target_tokens: int = TARGET_TOKENS) -> None:
    os.makedirs(DATA_DIR, exist_ok=True)

    if os.path.exists(RAW_TEXT_FILE):
        log.info("Raw text already exists at %s — delete it to re-download.", RAW_TEXT_FILE)
        return

    log.info("Streaming C4 (en) from HuggingFace …")
    dataset = load_dataset(
        "allenai/c4",
        "en",
        split="train",
        streaming=True,
        trust_remote_code=True,
    )

    total_tokens = 0
    total_docs   = 0
    log_interval = 1_000_000  # log every 1 M tokens
    next_log     = log_interval

    with open(RAW_TEXT_FILE, "w", encoding="utf-8") as fout:
        for example in dataset:
            text = example["text"].strip()
            if not text:
                continue

            # crude token count (whitespace split) — good enough for budgeting
            n_tok = len(text.split())
            total_tokens += n_tok
            total_docs   += 1

            fout.write(text.replace("\n", " ") + "\n")   # one doc per line

            if total_tokens >= next_log:
                log.info("  collected %d M tokens  (%d docs)", total_tokens // 1_000_000, total_docs)
                next_log += log_interval

            if total_tokens >= target_tokens:
                break

    log.info("Done.  %d documents  |  ~%d M whitespace tokens", total_docs, total_tokens // 1_000_000)
    log.info("Saved → %s", RAW_TEXT_FILE)


if __name__ == "__main__":
    download_c4()
