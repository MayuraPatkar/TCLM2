# TCLM2: Transformer-based Causal Language Model v2

**TCLM2** is a custom, decoder-only Transformer model trained using **causal language modeling (CLM)**. It is designed for autoregressive text generation tasks and supports fine-tuning and inference workflows similar to GPT-style models.

---

## 🧠 Overview

TCLM2 is built for fast, scalable, and high-quality text generation. It uses multi-layer self-attention with positional embeddings, LayerNorm, and residual connections, trained on a large corpus with left-to-right prediction objectives.

---

## 🚀 Features

- ⚡ Decoder-only Transformer with custom architecture
- 📖 Causal (autoregressive) language modeling
- 🔧 Hugging Face-compatible interface for loading, fine-tuning, and inference
- 🔬 Pre-trained using `Trainer` with dynamic masking and gradient checkpointing

---

## 📁 Project Structure

```bash
TCLM2/
├── model/                # Model architecture (config, forward, init)
├── tokenizer/            # Tokenizer setup and special tokens
├── training/             # Scripts for training and evaluation
├── generate.py           # Script for text generation
├── config.json           # Model configuration
├── pytorch_model.bin     # Model weights (optional)
└── README.md             # Documentation
