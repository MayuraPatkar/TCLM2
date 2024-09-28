import torch

def get_config():
    return {
        "batch_size": 1,
        "num_epochs": 50,
        "lr": 1e-4,
        "seq_len": 512,
        "d_model": 768,
        "n_layers": 12,
        "head": 12,
        "d_ff": 3072,
        "dropout": 0.1,
        "masking_prob": 0.15,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "model_file_path": "T-CLM2.pt",
        "tokenizer_file": "tokenizer.json",
    }