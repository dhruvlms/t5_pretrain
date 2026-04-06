"""
data/tokenizer_train.py
───────────────────────
Trains a SentencePiece Unigram tokenizer on c4_raw.txt.

Special tokens (matching T5 convention):
    <pad>        id = 0
    </s>         id = 1   (EOS)
    <unk>        id = 2
    <extra_id_0> id = 32027   (VOCAB_SIZE - 1 - 0)
    ...
    <extra_id_99>id = 31928

RAM usage by environment:
    Colab  (12.7 GB) → input_sentence_size = 1,000,000  (~3 GB peak RAM)
    Kaggle (13.0 GB) → input_sentence_size = 1,000,000  (~3 GB peak RAM)
    Local  (32 GB+)  → input_sentence_size = 3,000,000  (~7 GB peak RAM)

Run once (after download_c4.py):
    python data/tokenizer_train.py

Output:
    data_store/tokenizer/tokenizer.model
    data_store/tokenizer/tokenizer.vocab
"""

import os, sys, logging

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import RAW_TEXT_FILE, TOKENIZER_DIR, VOCAB_SIZE, NUM_SENTINEL_TOKENS, ENV

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s")
log = logging.getLogger(__name__)

try:
    import sentencepiece as spm
except ImportError:
    raise SystemExit("Run:  pip install sentencepiece")


# ── RAM-safe sentence limit per environment ───────────────────────────────────
# Original value was 5,000,000 which causes OOM on Colab (12.7 GB RAM limit).
# 1,000,000 sentences gives equally good tokenizer quality — T5's tokenizer
# was trained on far less. Peak RAM at 1M sentences ≈ 3 GB, well within limits.
def _sentence_limit() -> int:
    if ENV in ("colab", "kaggle"):
        return 1_000_000   # safe for 12-13 GB RAM limit
    else:
        return 3_000_000   # local machine with more RAM


def train_tokenizer() -> None:
    os.makedirs(TOKENIZER_DIR, exist_ok=True)
    model_prefix = os.path.join(TOKENIZER_DIR, "tokenizer")

    if os.path.exists(model_prefix + ".model"):
        log.info("Tokenizer already trained at %s — delete to retrain.", model_prefix)
        return

    if not os.path.exists(RAW_TEXT_FILE):
        raise FileNotFoundError(
            f"Raw text not found: {RAW_TEXT_FILE}\n"
            f"Run download_c4.py first."
        )

    file_size_gb = os.path.getsize(RAW_TEXT_FILE) / 1e9
    log.info("Input file : %s  (%.1f GB)", RAW_TEXT_FILE, file_size_gb)

    # SentencePiece trains on real_vocab pieces;
    # sentinel tokens are injected as user_defined_symbols.
    real_vocab   = VOCAB_SIZE - NUM_SENTINEL_TOKENS
    sentinels    = [f"<extra_id_{i}>" for i in range(NUM_SENTINEL_TOKENS)]
    sentence_cap = _sentence_limit()

    log.info(
        "Training SentencePiece tokenizer  "
        "(vocab=%d  sentinels=%d  max_sentences=%d  env=%s)",
        VOCAB_SIZE, NUM_SENTINEL_TOKENS, sentence_cap, ENV.upper(),
    )
    log.info("Expected RAM usage: ~%.1f GB", sentence_cap / 1_000_000 * 3.0)
    log.info("Expected time     : ~%d minutes", 10 if ENV in ("colab","kaggle") else 20)

    spm.SentencePieceTrainer.train(
        input                        = RAW_TEXT_FILE,
        model_prefix                 = model_prefix,
        model_type                   = "unigram",
        vocab_size                   = real_vocab,
        character_coverage           = 0.9995,
        pad_id                       = 0,
        eos_id                       = 1,
        unk_id                       = 2,
        bos_id                       = -1,           # T5 has no BOS
        pad_piece                    = "<pad>",
        eos_piece                    = "</s>",
        unk_piece                    = "<unk>",
        user_defined_symbols         = sentinels,
        input_sentence_size          = sentence_cap, # ← FIXED: was 5_000_000
        shuffle_input_sentence       = True,
        num_threads                  = min(4, os.cpu_count() or 2),
        train_extremely_large_corpus = True,
        # Null character handling — C4 has some null bytes, ignore them
        remove_extra_whitespaces     = True,
        normalization_rule_name      = "nmt_nfkc",
    )

    log.info("Tokenizer saved → %s.model", model_prefix)
    _verify(model_prefix + ".model")


def _verify(model_path: str) -> None:
    sp = spm.SentencePieceProcessor(model_file=model_path)
    sample  = "The quick brown fox jumps over the lazy dog."
    ids     = sp.encode(sample)
    decoded = sp.decode(ids)

    log.info("Verification")
    log.info("  input   : %r", sample)
    log.info("  encoded : %d tokens → %r", len(ids), ids[:10])
    log.info("  decoded : %r", decoded)
    log.info("  <pad>=0 : %s", sp.piece_to_id("<pad>") == 0)
    log.info("  </s>=1  : %s", sp.piece_to_id("</s>") == 1)
    log.info("  vocab   : %d", sp.get_piece_size())

    s0 = sp.piece_to_id("<extra_id_0>")
    log.info("  <extra_id_0> id = %d  (expected %d)", s0, VOCAB_SIZE - NUM_SENTINEL_TOKENS)
    assert s0 == VOCAB_SIZE - NUM_SENTINEL_TOKENS, (
        f"Sentinel ID mismatch: got {s0}, expected {VOCAB_SIZE - NUM_SENTINEL_TOKENS}"
    )
    log.info("Tokenizer verified ✅")


if __name__ == "__main__":
    train_tokenizer()
