"""
data/tokenizer_train.py
───────────────────────
Trains a SentencePiece Unigram tokenizer on c4_raw.txt.

Special tokens (matching T5 convention):
    <pad>        id = 0
    </s>         id = 1   (EOS)
    <unk>        id = 2
    <extra_id_0> id = 32027   (VOCAB_SIZE - 1 - 0)
    …
    <extra_id_99>id = 31928

Run once (after download_c4.py):
    python data/tokenizer_train.py

Output:
    data_store/tokenizer/tokenizer.model
    data_store/tokenizer/tokenizer.vocab
"""

import os, sys, logging

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import DATA_DIR, RAW_TEXT_FILE, TOKENIZER_DIR, VOCAB_SIZE, NUM_SENTINEL_TOKENS

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s")
log = logging.getLogger(__name__)

try:
    import sentencepiece as spm
except ImportError:
    raise SystemExit("Run:  pip install sentencepiece")


def train_tokenizer() -> None:
    os.makedirs(TOKENIZER_DIR, exist_ok=True)
    model_prefix = os.path.join(TOKENIZER_DIR, "tokenizer")

    if os.path.exists(model_prefix + ".model"):
        log.info("Tokenizer already trained at %s — delete to retrain.", model_prefix)
        return

    if not os.path.exists(RAW_TEXT_FILE):
        raise FileNotFoundError(f"Raw text not found: {RAW_TEXT_FILE}\nRun download_c4.py first.")

    # SentencePiece trains on <VOCAB_SIZE - NUM_SENTINEL_TOKENS> real pieces;
    # sentinel tokens are injected as user_defined_symbols.
    real_vocab = VOCAB_SIZE - NUM_SENTINEL_TOKENS
    sentinels  = [f"<extra_id_{i}>" for i in range(NUM_SENTINEL_TOKENS)]

    log.info("Training SentencePiece tokenizer  (vocab=%d, sentinels=%d) …", VOCAB_SIZE, NUM_SENTINEL_TOKENS)

    spm.SentencePieceTrainer.train(
        input                  = RAW_TEXT_FILE,
        model_prefix           = model_prefix,
        model_type             = "unigram",
        vocab_size             = real_vocab,
        character_coverage     = 0.9995,
        pad_id                 = 0,
        eos_id                 = 1,
        unk_id                 = 2,
        bos_id                 = -1,             # T5 has no BOS
        pad_piece              = "<pad>",
        eos_piece              = "</s>",
        unk_piece              = "<unk>",
        user_defined_symbols   = sentinels,      # appended after real vocab
        input_sentence_size    = 5_000_000,      # sample 5 M sentences for speed
        shuffle_input_sentence = True,
        num_threads            = min(16, os.cpu_count() or 4),
        train_extremely_large_corpus = True,
    )

    log.info("Tokenizer saved → %s.model", model_prefix)
    _verify(model_prefix + ".model")


def _verify(model_path: str) -> None:
    sp = spm.SentencePieceProcessor(model_file=model_path)
    sample = "The quick brown fox jumps over the lazy dog."
    ids = sp.encode(sample)
    decoded = sp.decode(ids)
    log.info("Verification  |  input: %r  →  %d ids  →  %r", sample, len(ids), decoded)
    log.info("  <pad>=0? %s   </s>=1? %s   vocab=%d",
             sp.piece_to_id("<pad>") == 0,
             sp.piece_to_id("</s>") == 1,
             sp.get_piece_size())
    # Sentinel check
    s0 = sp.piece_to_id("<extra_id_0>")
    log.info("  <extra_id_0> id = %d  (expected ~%d)", s0, VOCAB_SIZE - NUM_SENTINEL_TOKENS)


if __name__ == "__main__":
    train_tokenizer()
