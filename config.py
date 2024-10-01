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
        "temperature": 0.7,
        "top_k": 50,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "model_file_path": "T-CLM2.pt",
        "dataset_file_path": "dataset.json",
        "tokenizer_file_path": "tokenizer.json",
        "logs": "/T-CLM-logs",
    }