import os

import torch

from torch.utils.data import DataLoader
import datasets
from torch4x.register import REGISTRY

def evaluation_by_pg19(
    filepath: str,
    max_token_length:int,
    batch_size: int,
    tokenizer
):
    dataset = datasets.load_from_disk(filepath)['test']
    dataset = dataset.select_columns(['text'])
    def trun_func(examples):
        text = examples['text']
        tokens = tokenizer.encode(text)
        # txt_len = len(tokens)
        # split_text = []
        # for i in range(0, txt_len, max_token_length):
        #     split_text.append(tokens[i:i+max_token_length])
        #     if len(split_text[-1])< max_token_length:
        #         split_text[-1] = split_text[-1] + [tokenizer.eos_token_id]*(max_token_length - len(split_text[-1]))
        if len(tokens) < max_token_length:
            tokens = tokens + [tokenizer.eos_token_id]*(max_token_length - len(tokens))
        split_text = [tokens[:max_token_length]]
        return {'split_text': split_text}
    dataset = dataset.map(trun_func)
    subsamples = []
    for example in dataset["split_text"]:
        subsamples.extend(example)
    new_datasets = {'input_ids':subsamples, 'labels': subsamples}
    dataset = datasets.Dataset.from_dict(new_datasets)
    # loader = DataLoader(dataset, batch_size=batch_size)
    return dataset