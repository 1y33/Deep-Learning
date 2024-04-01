from torch.utils.data import DataLoader,Dataset
import torch
import torch.nn as nn
import tiktoken

class GPTDatasetV1(Dataset):
    def __init__(self, txt, tokenizer, max_length, stride):
        self.tokenizer = tokenizer
        self.inputs_ids = []
        self.target_ids = []

        token_ids = tokenizer.encode(txt)

        for i in range(0, len(token_ids) - max_length, stride):
            input_chunk = token_ids[i:i + max_length]
            target_chunk = token_ids[i + 1:i + max_length + 1]
            self.inputs_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))

    def __len__(self):
        return len(self.inputs_ids)

    def __getitem__(self, idx):
        return self.inputs_ids[idx], self.target_ids[idx]


from torch.utils.data import DataLoader


def create_dataloader(txt, batch_size=4, max_length=256, stride=128, shuffle=True):
    toke = tiktoken.get_encoding("gpt2")
    dataset = GPTDatasetV1(txt, toke, max_length, stride)
    dataloader = DataLoader(dataset, batch_size, shuffle=shuffle)

    return dataloader