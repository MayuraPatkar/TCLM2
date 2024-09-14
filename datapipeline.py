import torch
from torch.utils.data import DataLoader, random_split
import json
from tokenizer import build_or_get_tokenizer
from torch.utils.data import Dataset

from datasets import load_dataset
from config import get_config

class BilingualDataset(Dataset):
    def __init__(self, ds, tokenizer, seq_len):
        self.seq_len = seq_len
        self.ds = ds
        self.tokenizer = tokenizer
        self.sos_token = torch.tensor([tokenizer.token_to_id("[SOS]")], dtype=torch.int64)
        self.eos_token = torch.tensor([tokenizer.token_to_id("[EOS]")], dtype=torch.int64)
        self.pad_token = torch.tensor([tokenizer.token_to_id("[PAD]")], dtype=torch.int64)

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        text = self.ds[idx]['text']
        input_tokens = self.tokenizer.encode(text).ids

        # Truncate if too long
        if len(input_tokens) > self.seq_len - 2:
            input_tokens = input_tokens[:self.seq_len - 2]

        num_padding_tokens = self.seq_len - len(input_tokens) - 2

        encoder_input = torch.cat(
            [
                self.sos_token,
                torch.tensor(input_tokens, dtype=torch.int64),
                self.eos_token,
                torch.tensor([self.pad_token] * num_padding_tokens, dtype=torch.int64),
            ],
            dim=0,
        )

        # The label should also be of length seq_len
        label = torch.cat(
            [
                torch.tensor(input_tokens, dtype=torch.int64),
                self.eos_token,
                torch.tensor([self.pad_token] * (num_padding_tokens + 1), dtype=torch.int64),  # Adjusted padding
            ],
            dim=0,
        )

        # Debug prints
        print(f"Text: {text}")
        print(f"Input tokens length: {len(input_tokens)}")
        print(f"Num padding tokens: {num_padding_tokens}")
        print(f"Encoder input size: {encoder_input.size(0)}")
        print(f"Label size: {label.size(0)}")
        print("="*50)

        assert encoder_input.size(0) == self.seq_len
        assert label.size(0) == self.seq_len

        return {
            "encoder_input": encoder_input,
            "label": label,
            "encoder_mask": (encoder_input != self.pad_token).unsqueeze(0).unsqueeze(0).int(),
            "text": text,
        }



def causal_mask(size):
    mask = torch.triu(torch.ones((1, size, size)), diagonal=1).type(torch.int)
    return mask == 0

def get_ds(config):
    # with open('dataset-mini.json', 'r', encoding='utf-8') as f:
    #     ds_raw = json.load(f)

    ds_raw = load_dataset(f"bookcorpus/bookcorpus", f"plain_text", split='train', trust_remote_code=True)

    tokenizer = build_or_get_tokenizer(config, ds_raw)
    train_ds_size = int(0.95 * len(ds_raw))
    val_ds_size = len(ds_raw) - train_ds_size
    train_ds_raw, val_ds_raw = random_split(ds_raw, [train_ds_size, val_ds_size])

    train_ds = BilingualDataset(train_ds_raw, tokenizer, config['seq_len'])
    val_ds = BilingualDataset(val_ds_raw, tokenizer, config['seq_len'])

    train_dataloader = DataLoader(train_ds, batch_size=config['batch_size'], shuffle=True)
    val_dataloader = DataLoader(val_ds, batch_size=1, shuffle=True)

    return train_dataloader, val_dataloader, tokenizer


# if __name__ == '__main__':
#     config = get_config()
#     get_ds(config)