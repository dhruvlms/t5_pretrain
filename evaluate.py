"""
evaluate.py
───────────
Load a saved checkpoint and evaluate on the test set.
Also runs a qualitative span-infilling demo.

Usage:
    python evaluate.py --checkpoint checkpoints/best_model.pt
    python evaluate.py --checkpoint checkpoints/best_model.pt --demo
"""

import argparse, os, sys, logging
import torch

from config import MODEL_CFG, TRAIN_CFG, TOKENIZER_DIR
from model.transformer import T5Model, count_parameters
from data.dataset import make_dataloaders
from training.callbacks import CheckpointCallback
from training.trainer import Trainer

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s")
log = logging.getLogger(__name__)


def load_tokenizer():
    model_path = os.path.join(TOKENIZER_DIR, "tokenizer.model")
    import sentencepiece as spm
    sp = spm.SentencePieceProcessor(model_file=model_path)
    return sp


def run_demo(model, sp, device, n_examples: int = 3):
    """Run a quick qualitative span-infilling demo."""
    from data.dataset import apply_span_corruption, pad_or_truncate
    from config import MAX_SEQ_LEN, MAX_TARGET_LEN, VOCAB_SIZE, NUM_SENTINEL_TOKENS

    import numpy as np

    sentences = [
        "The transformer architecture has revolutionized natural language processing tasks.",
        "Deep learning models require large amounts of training data to generalize well.",
        "Attention mechanisms allow models to focus on relevant parts of the input sequence.",
    ]

    model.eval()
    for sentence in sentences[:n_examples]:
        tokens = sp.encode(sentence, out_type=int)
        tok_arr = np.array(tokens, dtype=np.int32)

        enc_ids, dec_ids = apply_span_corruption(tok_arr)
        enc_ids, enc_mask = pad_or_truncate(enc_ids, MAX_SEQ_LEN)

        enc_tensor  = torch.tensor([enc_ids],  dtype=torch.long, device=device)
        mask_tensor = torch.tensor([enc_mask], dtype=torch.long, device=device)

        pred_ids = model.generate(enc_tensor, mask_tensor, max_new_tokens=32)
        pred_ids = pred_ids[0].tolist()
        # Strip sentinel tokens and EOS
        clean = [t for t in pred_ids if t not in (1,) and t < VOCAB_SIZE - NUM_SENTINEL_TOKENS]
        pred_text = sp.decode(clean)

        corrupted_text = sp.decode([t for t in enc_ids if t > 0 and t < VOCAB_SIZE - NUM_SENTINEL_TOKENS])

        print(f"\nOriginal  : {sentence}")
        print(f"Corrupted : {corrupted_text}")
        print(f"Predicted : {pred_text}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--demo",       action="store_true", help="Run qualitative demo after eval")
    p.add_argument("--max_batches",type=int, default=None, help="Limit test batches (None=full)")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("Device: %s", device)

    # ── Load model ────────────────────────────────────────────────────────────
    model = T5Model(MODEL_CFG).to(device)
    CheckpointCallback.load(args.checkpoint, model, device=device)
    log.info("Parameters: %d  (%.1f M)", count_parameters(model), count_parameters(model) / 1e6)

    # ── Evaluate ─────────────────────────────────────────────────────────────
    _, _, test_dl = make_dataloaders()

    trainer = Trainer.__new__(Trainer)
    trainer.model    = model
    trainer.device   = device
    trainer.cfg      = TRAIN_CFG
    trainer.global_step = 0

    test_loss = trainer._evaluate(test_dl, max_batches=args.max_batches)
    import math
    ppl = math.exp(min(test_loss, 20))
    log.info("Test loss: %.4f   Test perplexity: %.2f", test_loss, ppl)

    # ── Demo ──────────────────────────────────────────────────────────────────
    if args.demo:
        try:
            sp = load_tokenizer()
            log.info("\n── Span Infilling Demo ──────────────────────────────────")
            run_demo(model, sp, device)
        except Exception as e:
            log.warning("Demo failed: %s", e)


if __name__ == "__main__":
    main()
