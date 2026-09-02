# MiniGPT

A minimal GPT language model from scratch in PyTorch. Character-level tokenization, stacked causal transformer decoder blocks, autoregressive text generation. Trained on tinyshakespeare.

## Architecture

```
Input Token IDs
      │
Token Embedding + Position Embedding
      │
   Dropout
      │
┌─────────────────┐
│ Transformer Block│ × N layers
│  ├─ Causal MHA  │
│  ├─ Add & Norm  │
│  ├─ FFN         │
│  └─ Add & Norm  │
└─────────────────┘
      │
  LayerNorm
      │
  LM Head (linear → vocab_size)
      │
  Logits → Cross-Entropy Loss
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run Tests

```bash
pytest tests/ -v
```
