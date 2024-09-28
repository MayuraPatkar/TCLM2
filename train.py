import torch
import warnings
import sys
from tqdm import tqdm
from config import get_config
from datapipeline import get_ds
from loadmodel import load_model
from model import TCLM
# import wandb

def train(config):
    # # Initialize wandb
    # wandb.init(project="T-CLM2", config=config)

    device = config['device']
    print("Using device:", device)
    if device == 'cuda':
        print(f"Device name: {torch.cuda.get_device_name(device.index)}")
        print(f"Device memory: {round(torch.cuda.get_device_properties(device.index).total_memory / 1024 ** 3, 1)} GB")
    device = torch.device(device)

    train_dataloader, val_dataloader, tokenizer = get_ds(config)
    model = TCLM(vocab_size=tokenizer.get_vocab_size(), seq_len=config['seq_len'], d_model=config['d_model'], N=config['n_layers'], h=config['head'], dropout=config['dropout'], d_ff=config['d_ff'])

    # Log model configuration in wandb
    # wandb.watch(model, log="all")

    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'], eps=1e-9)
    model, initial_epoch, global_step = load_model(config, device, model, tokenizer, optimizer)

    for epoch in range(initial_epoch, config['num_epochs']):
        torch.cuda.empty_cache()
        model.train()
        batch_iterator = tqdm(train_dataloader, desc=f"Processing Epoch {epoch:02d}")
        total_loss = 0
        num_batches = len(train_dataloader)

        for batch in batch_iterator:
            encoder_input = batch['input'].to(device)
            targets = batch['label'].to(device)

            optimizer.zero_grad()  # Reset gradients
            logits, loss = model(encoder_input, targets=targets)

            total_loss += loss.item()  # Accumulate loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # Gradient clipping
            optimizer.step()

            global_step += 1

            # Log batch loss to wandb
            # wandb.log({"batch_loss": loss.item(), "global_step": global_step})

            batch_iterator.set_postfix({'Loss': loss.item()})

        # Epoch-level logging
        avg_loss = total_loss / num_batches
        print(f"Epoch {epoch} | Avg Loss: {round(avg_loss, 2)}")

        # Log average loss for epoch to wandb
        # wandb.log({"average_loss": avg_loss, "epoch": epoch})

        # Model checkpointing
        model_filename = f"T-CLM2.pt"
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'global_step': global_step,
        }, model_filename)

        validate(model, val_dataloader, device, epoch)

def validate(model, val_dataloader, device, epoch):
    model.eval()
    total_val_loss = 0
    num_batches = len(val_dataloader)

    with torch.no_grad():
        for batch in val_dataloader:
            encoder_input = batch['input'].to(device)
            targets = batch['label'].to(device)
            logits, val_loss = model(encoder_input, targets=targets)
            total_val_loss += val_loss.item()

    avg_val_loss = total_val_loss / num_batches
    print(f"Validation Loss (Epoch {epoch}): {avg_val_loss}")

    # Log validation loss to wandb
    # wandb.log({"validation_loss": avg_val_loss, "epoch": epoch})

if __name__ == '__main__':
    warnings.filterwarnings("ignore")
    config = get_config()
    train(config)
