def get_config():
    return {
        "experiment_name": "T-CLM",
        "batch_size": 1,
        "num_epochs": 50,
        "lr": 5e-5,
        "seq_len": 512,
        "d_model": 768,
        "n_layers": 12,
        "head": 12,
        "d_ff": 3072,
        "dropout": 0.1,
        "masking_prob": 0.15,
        "model_file_path": "T-CLM.pt",
        "tokenizer_file": "tokenizer.json",
    }