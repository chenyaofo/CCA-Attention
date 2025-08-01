import os

import torch

from torch.utils.data import DataLoader
from torchdata.datapipes.iter import FileLister, FileOpener, TFRecordLoader, Mapper, Shuffler, Batcher, Collator, ShardingFilter
from torch4x.register import REGISTRY
import datasets
from transformers.models.llama.modeling_llama import LlamaForCausalLM


def pretraining_slimpajamar_dataset(
    filepath: str,
    seed: int,
):
    dataset = datasets.load_dataset(filepath, split="train")
    dataset = dataset.select_columns(['input_ids', 'labels'])
    dataset = dataset.shuffle(seed=seed)
    # loader = DataLoader(dataset, batch_size=batch_size)
    return dataset

def pretraining_slimpajamar_llama3(file_path: str, seed: int):
    print(f'------load llam3 dataset from {file_path}')
    dataset = datasets.load_dataset('json', data_files=file_path, split='train')
    dataset = dataset.select_columns(['input_ids', 'labels'])
    dataset = dataset.shuffle(seed=seed)
    return dataset

    

def collate_truncate(batch, max_token_length, enc_tokenizer=None, dec_tokenizer=None):
    input_ids=[]
    labels=[]
    for example in batch:
        example['input_ids'] = example['input_ids'][:max_token_length]
        input_ids.append(example['input_ids'])
        labels.append([example['labels']])
    new_batch = {
        'input_ids': torch.tensor(input_ids),
        'labels': torch.tensor(input_ids)
    }
    return new_batch
