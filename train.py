import torch
import torch.nn as nn
import warnings
import os
import sys
from pathlib import Path
from model import build_transformer
from config import get_config
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from datapipeline import get_ds, causal_mask

def greedy_decode(model, source, source_mask, tokenizer, max_len, device):
    sos_idx = tokenizer.token_to_id('[SOS]')
    eos_idx = tokenizer.token_to_id('[EOS]')

    encoder_output = model.encode(source, source_mask)
    decoder_input = torch.empty(1, 1).fill_(sos_idx).type_as(source).to(device)
    while True:
        if decoder_input.size(1) == max_len:
            break

        decoder_mask = causal_mask(decoder_input.size(1)).type_as(source_mask).to(device)
        out = model.decode(encoder_output, source_mask, decoder_input, decoder_mask)

        prob = model.project(out[:, -1])
        _, next_word = torch.max(prob, dim=1)
        decoder_input = torch.cat(
            [decoder_input, torch.empty(1, 1).type_as(source).fill_(next_word.item()).to(device)], dim=1
        )

        if next_word == eos_idx:
            break

    return decoder_input.squeeze(0)


def run_validation(model, validation_ds, tokenizer, max_len, device, print_msg, global_step, writer, num_examples=2):
    model.eval()
    count = 0

    source_texts = []
    predicted = []

    console_width = 80

    with torch.no_grad():
        for batch in validation_ds:
            count += 1
            encoder_input = batch["encoder_input"].to(device) # (b, seq_len)
            encoder_mask = batch["encoder_mask"].to(device) # (b, 1, 1, seq_len)

            assert encoder_input.size(0) == 1, "Batch size must be 1 for validation"

            model_out = greedy_decode(model, encoder_input, encoder_mask, tokenizer, max_len, device)

            source_text = batch["text"][0]
            model_out_text = tokenizer.decode(model_out.detach().cpu().numpy())

            source_texts.append(source_text)
            predicted.append(model_out_text)
            
            print_msg('-'*console_width)
            print_msg(f"{f'SOURCE: ':>12}{source_text}")
            print_msg(f"{f'model out: ':>12}{model_out}")
            print_msg(f"{f'PREDICTED: ':>12}{model_out_text}")

            if count == num_examples:
                print_msg('-'*console_width)
                break

def get_weights_file_path(config):
    model_file_path = config.get('model_file_path', '')
    if Path(model_file_path).exists():
        return str(model_file_path)
    else:
        return None

def get_model(config, vocab_len):
    model = build_transformer(vocab_len, vocab_len, config["seq_len"], config['seq_len'], d_model=config['d_model'],N=config['n_layers'],h=config['head'],dropout=config['dropout'],d_ff=config['d_ff'])
    return model

def train(config):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)
    if (device == 'cuda'):
        print(f"Device name: {torch.cuda.get_device_name(device.index)}")
        print(f"Device memory: {torch.cuda.get_device_properties(device.index).total_memory / 1024 ** 3} GB")
    device = torch.device(device)

    train_dataloader, val_dataloader, tokenizer = get_ds(config)

    model = get_model(config, tokenizer.get_vocab_size()).to(device)
    # Tensorboard
    writer = SummaryWriter(config['experiment_name'])

    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'], eps=1e-9)

    initial_epoch = 0
    global_step = 0
    model_filename = get_weights_file_path(config)
    if model_filename:
        print(f'Preloading model {model_filename}')
        state = torch.load(model_filename)
        model.load_state_dict(state['model_state_dict'])
        initial_epoch = state['epoch'] + 1
        optimizer.load_state_dict(state['optimizer_state_dict'])
        global_step = state['global_step']
    else:
        print('No model to preload, starting from scratch')

    loss_fn = nn.CrossEntropyLoss(ignore_index=tokenizer.token_to_id('[PAD]'), label_smoothing=0.1).to(device)

    for epoch in range(initial_epoch, config['num_epochs']):
        torch.cuda.empty_cache()
        model.train()
        batch_iterator = tqdm(train_dataloader, desc=f"Processing Epoch {epoch:02d}")
        total_loss = 0
        total_accuracy = 0
        num_batches = len(train_dataloader)

        for batch in batch_iterator:
            encoder_input = batch['encoder_input'].to(device) # (b, seq_len)
            decoder_input = batch['label'].to(device) # (B, seq_len)
            encoder_mask = batch['encoder_mask'].to(device) # (B, 1, 1, seq_len)
            decoder_mask = batch['encoder_mask'].to(device) # (B, 1, seq_len, seq_len)

            encoder_output = model.encode(encoder_input, encoder_mask)
            decoder_output = model.decode(encoder_output, encoder_mask, decoder_input, decoder_mask)
            proj_output = model.project(decoder_output)

            label = batch['label'].to(device)

            loss = loss_fn(proj_output.view(-1, tokenizer.get_vocab_size()), label.view(-1))
            total_loss += loss.item()

            loss.backward()

            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            
        print(f"Totoal Loss: {total_loss}")
        global_step += 1
        # Log the loss
        writer.add_scalar('train loss', total_loss, global_step)
        writer.flush()

        avg_loss = total_loss / num_batches
        writer.add_scalar('train average loss', avg_loss, epoch)

        run_validation(model, val_dataloader, tokenizer, config['seq_len'], device, lambda msg: batch_iterator.write(msg), global_step, writer)


        model_filename = "T-CLM.pt"
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'global_step': global_step,
        }, model_filename)


if __name__ == '__main__':
    warnings.filterwarnings("ignore")
    config = get_config()
    train(config)