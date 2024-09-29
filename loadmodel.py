import torch
from pathlib import Path

def load_model(config, device, model, optimizer):
    initial_epoch = 0
    global_step = 0
    model = model.to(device)
    model_file_path = config['model_file_path']
    if Path(model_file_path).exists():
        print(f'Loading model from {str(model_file_path)}')
        state = torch.load(str(model_file_path), map_location=device, weights_only=True)
        model.load_state_dict(state['model_state_dict'])
        initial_epoch = state['epoch'] + 1
        optimizer.load_state_dict(state['optimizer_state_dict'])
        global_step = state['global_step']
    else:
        print("No model file found.")

    return model, initial_epoch, global_step