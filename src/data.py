import logging
import os

import pandas as pd
import pytorch_lightning as pl
import sklearn.model_selection
import torch
from transformers import AutoTokenizer
from torch.utils.data import DataLoader, Dataset

import utils


class WikipediaDataset(Dataset):
    def __init__(self, examples, wndata, training):
        self.examples = examples
        self.wndata = wndata
        self.training = training

    def __len__(self):
        return len(self.examples) * 2

    def __getitem__(self, item):
        ex = self.examples[item // 2]
        subj, verb, obj, negative_subj, negative_verb, negative_obj = ex[:6]
        subj_hyp, obj_hyp, neg_subj_hyp, neg_obj_hyp = ex[6:]

        if item % 2 == 0:
            return self.wndata.item_to_features(subj, verb, obj,
                                                subj_hyp, obj_hyp, 1, self.training)
        else:
            return self.wndata.item_to_features(negative_subj, negative_verb, negative_obj,
                                                neg_subj_hyp, neg_obj_hyp, 0, self.training)


class PlausibilityDataset(Dataset):
    def __init__(self, fname, wndata):
        self.examples = torch.load(fname)
        self.wndata = wndata

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, item):
        ex = self.examples[item]
        label, subj, verb, obj, subj_hyp, obj_hyp = ex
        
        return self.wndata.item_to_features(subj, verb, obj,
                                            subj_hyp, obj_hyp, int(label), False)


class DataModule(pl.LightningDataModule):
    def __init__(
        self,
        model_type: str,
        model_name_or_path: str,
        cache_dir: str,
        data_dir: str,
        max_seq_length: int = 16,
        train_batch_size: int = 128,
        eval_batch_size: int = 32,
        **kwargs,
    ):
        super().__init__()
        self.model_type = model_type
        self.model_name_or_path = model_name_or_path
        self.max_seq_length = max_seq_length
        self.train_batch_size = train_batch_size
        self.eval_batch_size = eval_batch_size
        self.cache_dir = cache_dir
        self.data_dir = data_dir

    def prepare_data(self):
        """ Parse, preprocess, and cache the datasets. """
        self.prepare_wikipedia_data()
        self.prepare_plausibility_data()
        utils.WordNetData(self.data_dir, self.cache_dir,
                          self.model_type, self.model_name_or_path)

    def setup(self, stage: str):
        logging.info('Setting up data module')

        wndata = utils.WordNetData(self.data_dir, self.cache_dir,
                                   self.model_type, self.model_name_or_path)

        train_fname = os.path.join(self.cache_dir, 'train.pt')
        val_fname   = os.path.join(self.cache_dir, 'val.pt')
        
        pep_3k_fname          = os.path.join(self.cache_dir, 'pep_3k_valid.pt')
        twentyquestions_fname = os.path.join(self.cache_dir, 'twentyquestions_valid.pt')

        self.dataset = {}
        self.dataset['train'] = WikipediaDataset(torch.load(train_fname), wndata, True)
        self.dataset['val'] = WikipediaDataset(torch.load(val_fname), wndata, False)
        
        self.dataset['pep_3k_valid'] = PlausibilityDataset(pep_3k_fname, wndata)
        self.dataset['twentyquestions_valid'] = PlausibilityDataset(twentyquestions_fname, wndata)

    def train_dataloader(self):
        return DataLoader(
            self.dataset['train'],
            batch_size=self.train_batch_size,
            shuffle=True,
            num_workers=10,
            collate_fn=utils.collate_batch
        )

    def val_dataloader(self):
        return [
                    DataLoader(self.dataset['val'],
                               batch_size=self.eval_batch_size,
                               collate_fn=utils.collate_batch),
                    DataLoader(self.dataset['pep_3k_valid'],
                               batch_size=self.eval_batch_size,
                               collate_fn=utils.collate_batch),
                    DataLoader(self.dataset['twentyquestions_valid'],
                               batch_size=self.eval_batch_size,
                               collate_fn=utils.collate_batch),
        ]

    def prepare_wikipedia_data(self):
        """ Parse, preprocess, and cache the Wikipedia training data. """
        train_fname = os.path.join(self.cache_dir, 'train.pt')
        val_fname   = os.path.join(self.cache_dir, 'val.pt')
        if os.path.exists(train_fname) and os.path.exists(val_fname):
            return

        logging.info('Preprocessing Wikipedia data.')

        fname = os.path.join(self.data_dir, 'english_wikipedia', 'train.parquet')
        df = pd.read_parquet(fname)
        train_df, val_df = sklearn.model_selection.train_test_split(df,
                                                                    test_size=1000,
                                                                    random_state=0)

        utils.cache_df(train_df, train_fname)
        utils.cache_df(val_df, val_fname)

    def prepare_plausibility_data(self):
        for dataset in ['pep_3k', 'twentyquestions']:
            for split in ['valid', 'test']:
                self.prepare_plausibility_dataset(dataset, split)

    def prepare_plausibility_dataset(self, dataset, split):
        cache_fname = os.path.join(self.cache_dir, f'{dataset}_{split}.pt')
        if os.path.exists(cache_fname):
            return
        
        data_fname   = os.path.join(self.data_dir, dataset, f'{split}.tsv')
        senses_fname = os.path.join(self.data_dir, dataset, f'{split}_senses.tsv')
        examples = utils.read_tsv(data_fname)
        senses   = utils.read_tsv(senses_fname)
        data = [ex + sense for ex, sense in zip(examples, senses)]
        
        torch.save(data, cache_fname)
