from safetensors.torch import save_file
import torch
from config import get_config
from datapipeline import get_ds
from model import TCLM

config = get_config()
_, _, tokenizer = get_ds(config)

model = TCLM(vocab_size=tokenizer.get_vocab_size(),
             seq_len=config['seq_len'],
             d_model=config['d_model'],
             N=config['n_layers'],
             h=config['head'],
             dropout=config['dropout'],
             d_ff=config['d_ff'])

# Get model's state dictionary (parameters)
model_state_dict = model.state_dict()

model_state_dict = {key: value.detach().clone() for key, value in model.state_dict().items()}

# Save the state dictionary using safetensors
save_file(model_state_dict, "model.safetensors")
