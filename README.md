# T5-Style Transformer Pretraining (~60M Parameters)
## Pipeline: C4 Dataset → Span Corruption → Encoder-Decoder Training

### Project Structure
```
t5_pretrain/
├── README.md
├── config.py              # All hyperparameters & paths
├── data/
│   ├── download_c4.py     # Download 300M tokens of C4
│   ├── tokenizer_train.py # Train SentencePiece tokenizer
│   ├── dataset.py         # Dataset + DataLoader + span corruption
│   └── preprocess.py      # Tokenize entire C4 → disk
├── model/
│   ├── transformer.py     # Full Encoder-Decoder Transformer
│   ├── attention.py       # Multi-head attention
│   └── feedforward.py     # FFN + RMSNorm
├── training/
│   ├── trainer.py         # Main training loop
│   ├── optimizer.py       # Adafactor optimizer
│   ├── scheduler.py       # LR scheduler
│   └── callbacks.py       # Checkpointing, early stopping, logging
├── utils/
│   ├── metrics.py         # Loss, perplexity tracking
│   └── logger.py          # Structured logging
├── train.py               # ENTRY POINT — run this
└── evaluate.py            # Evaluate saved checkpoint
```

### Quick Start
```bash
# 1. Install dependencies
pip install torch sentencepiece datasets transformers tqdm wandb numpy

# 2. Download C4 + train tokenizer + preprocess (one-time setup)
python data/download_c4.py       # ~2-3 hrs, downloads ~300M tokens
python data/tokenizer_train.py   # ~30 min
python data/preprocess.py        # ~1-2 hrs, tokenizes everything

# 3. Start training
python train.py

# 4. Evaluate best checkpoint
python evaluate.py --checkpoint checkpoints/best_model.pt
```

### Hardware Requirements
- GPU: 1x A100 (40GB) recommended, or 2x 3090 (24GB)
- RAM: 32GB+
- Disk: ~50GB for data + checkpoints

### Monitoring
- TensorBoard: `tensorboard --logdir logs/`
- WandB: set WANDB_API_KEY in environment
