import torch
from config import get_config
from loadmodel import load_model
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

optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'], eps=1e-9)
model, initial_epoch, global_step = load_model(config, config['device'], model, optimizer)

while True:
    text = input("text: ")

    if text == "exit" or text == "":
        break

    idx = tokenizer.encode(text).ids
    idx = torch.tensor([idx]).to(config['device'])
    generated_sequence = model.generate(idx, max_new_tokens=10, seq_len=config['seq_len'], temperature=config['temperature'], top_k=config['top_k'])
    predicted_text = tokenizer.decode(generated_sequence[0].cpu().numpy())
    print("predicted:", predicted_text)
