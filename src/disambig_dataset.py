from datetime import datetime
from typing import Optional

import logging

import torch
import pandas as pd
import pytorch_lightning as pl

from torch.utils.data import Dataset

from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from transformers import AutoTokenizer


class GenericDataset(Dataset):
    def __init__(self, examples):
        self.examples = examples

    def __len__(self):
        return len(self.examples['input_ids'])

    def __getitem__(self, item):
        return {
            "input_ids": self.examples["input_ids"][item],
            "attention_mask": self.examples["attention_mask"][item],
            "labels": self.examples["labels"][item],
            }


class DataModule(pl.LightningDataModule):
    def __init__(
        self,
        model_name_or_path: str,
        max_seq_length: int = 128,
        train_batch_size: int = 32,
        eval_batch_size: int = 32,
        **kwargs,
    ):
        super().__init__()
        self.model_name_or_path = model_name_or_path
        self.max_seq_length = max_seq_length
        self.train_batch_size = train_batch_size
        self.eval_batch_size = eval_batch_size
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name_or_path, use_fast=True)

    def setup(self, stage: str):
        print("Setting up data module")

        df = pd.read_parquet('')

        examples = []
        for index, row in df.iterrows():
            examples.append((index, row))
            if index > 100000:
                break
        
        examples_train, examples_val = train_test_split(examples, test_size=1000, random_state=0)

        self.dataset = {}
        self.dataset['train'] = GenericDataset(self.convert_to_features(examples_train))
        self.dataset['validation'] = GenericDataset(self.convert_to_features(examples_val))


    def train_dataloader(self):
        return DataLoader(
            self.dataset['train'],
            batch_size=self.train_batch_size,
            shuffle=True,
            num_workers=10
        )

    def val_dataloader(self):
        return DataLoader(
            self.dataset['validation'], 
            batch_size=self.eval_batch_size, 
            num_workers=10
            )

    def convert_to_features(self, examples):
        texts = []
        labels = []
        for ex_id, ex in examples:
            subj, verb, obj, neg_subj, _, neg_obj, _, _, _, _ = ex
            texts.append(f'{subj} {verb} {obj}')
            texts.append(f'{neg_subj} {verb} {neg_obj}')
            labels.append([1, 0])
            labels.append([0, 1])

        features = self.tokenizer.batch_encode_plus(
            texts, max_length=self.max_seq_length, padding='max_length', truncation=True,
            return_tensors='pt'
        )

        output = {}
        output['input_ids'] = features['input_ids']
        output['attention_mask'] = features['attention_mask']
        output['labels'] = torch.tensor(labels, dtype=torch.float)

        return output
