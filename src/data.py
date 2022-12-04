import logging
import os
from datetime import datetime
from typing import Optional

import pandas as pd
import pytorch_lightning as pl
import sklearn.model_selection
import torch
import transformers
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

import utils

tqdm.pandas()


class WikipediaDataset(Dataset):
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


class PlausibilityDataset(Dataset):
    def __init__(self, fname, tokenizer):
        self.examples = utils.read_tsv(fname)
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, item):
        ex = self.examples[item]
        label, subj, verb, obj = ex

        texts = [f'{subj} {verb} {obj}']
        
        labels = []
        if label == '1':
            labels.append([0, 1])
        else:
            labels.append([1, 0])

        features = self.tokenizer.batch_encode_plus(
            texts, max_length=16, padding='max_length', truncation=True,
            return_tensors='pt'
        )

        output = {}
        output['input_ids'] = features['input_ids'][0]
        output['attention_mask'] = features['attention_mask'][0]
        output['labels'] = torch.tensor(labels, dtype=torch.float)[0]

        return output


class DataModule(pl.LightningDataModule):
    def __init__(
        self,
        model_name_or_path: str,
        cache_dir: str,
        data_dir: str,
        max_seq_length: int = 16,
        train_batch_size: int = 128,
        eval_batch_size: int = 32,
        **kwargs,
    ):
        super().__init__()
        self.model_name_or_path = model_name_or_path
        self.max_seq_length = max_seq_length
        self.train_batch_size = train_batch_size
        self.eval_batch_size = eval_batch_size
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(self.model_name_or_path,
                                                                    use_fast=True)
        self.cache_dir = cache_dir
        self.data_dir = data_dir

    def prepare_data(self):
        """ Parse, preprocess, and cache the datasets. """
        self.prepare_wikipedia_data()

    def setup(self, stage: str):
        print("Setting up data module")

        train_fname = os.path.join(self.cache_dir, 'train.pt')
        val_fname   = os.path.join(self.cache_dir, 'val.pt')

        self.dataset = {}
        self.dataset['train'] = WikipediaDataset(torch.load(train_fname))
        self.dataset['validation'] = WikipediaDataset(torch.load(val_fname))


    def train_dataloader(self):
        return DataLoader(
            self.dataset['train'],
            batch_size=self.train_batch_size,
            shuffle=True
        )

    def val_dataloader(self):
        pep_3k_dir = os.path.join(self.data_dir, 'pep_3k', 'valid.tsv')
        twentyquestions_dir = os.path.join(self.data_dir, 'twentyquestions', 'valid.tsv')
        return [
                    DataLoader(self.dataset['validation'], batch_size=self.eval_batch_size),
                    DataLoader(PlausibilityDataset(pep_3k_dir, self.tokenizer),
                                batch_size=self.eval_batch_size),
                    DataLoader(PlausibilityDataset(twentyquestions_dir, self.tokenizer),
                                batch_size=self.eval_batch_size),
        ]

    def prepare_wikipedia_data(self):
        """ Parse, preprocess, and cache the Wikipedia training data. """
        train_fname = os.path.join(self.cache_dir, 'train.pt')
        val_fname   = os.path.join(self.cache_dir, 'val.pt')
        if os.path.exists(train_fname) and os.path.exists(val_fname):
            return
            
        logging.info("Preprocessing Wikipedia data.")

        fname = os.path.join(self.data_dir, 'english_wikipedia', 'train.parquet')
        df = pd.read_parquet(fname)
        train_df, val_df = sklearn.model_selection.train_test_split(df,
                                                                    train_size=1000000,
                                                                    test_size=1000,
                                                                    random_state=0)

        self.tokenize_and_cache_df(train_df, train_fname)
        self.tokenize_and_cache_df(val_df, val_fname)

    def tokenize_and_cache_df(self, df, fname):
        logging.info('Merging columns of dataframe.')
        df['svo'] = df[['subject', 'verb', 'object']].progress_apply(' '.join, axis=1)
        df['neg_svo'] = df[['negative_subject', 'verb', 'negative_object']].progress_apply(' '.join, axis=1)

        logging.info('Tokenizing texts.')
        texts = df['svo'].tolist() + df['neg_svo'].tolist()
        features = self.convert_to_features(texts)

        num_rows = df.shape[0]
        labels = torch.zeros((num_rows * 2, 2), dtype=torch.float)
        labels[:num_rows,1] = 1.0
        labels[num_rows:,0] = 1.0
        features['labels'] = labels

        logging.info('Saving cached data to: %s', fname)
        torch.save(features, fname)


    def convert_to_features(self, texts):
        features = self.tokenizer.batch_encode_plus(
            texts, max_length=self.max_seq_length, padding='max_length', truncation=True,
            return_tensors='pt'
        )
        return features
