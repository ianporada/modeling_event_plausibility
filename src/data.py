import logging
import os

import pandas as pd
import pytorch_lightning as pl
import sklearn.model_selection
import torch
from torch.utils.data import DataLoader, Dataset

import utils
            
            
class WikipediaDataset(Dataset):
    """Dataset for Wikipedia training data."""
    
    def __init__(self, examples, preprocessor, training):
        self.examples = examples
        self.preprocessor = preprocessor
        self.training = training

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, item):
        ex = self.examples[item]
        subj, verb, obj, negative_subj, negative_verb, negative_obj = ex[:6]
        subj_hyp, obj_hyp, neg_subj_hyp, neg_obj_hyp = ex[6:]
        # put a positive and negative example in each batch
        pos = self.preprocessor.item_to_features(subj, verb, obj,
                                                 subj_hyp, obj_hyp, 1, self.training)
        neg = self.preprocessor.item_to_features(negative_subj, negative_verb, negative_obj,
                                                 neg_subj_hyp, neg_obj_hyp, 0, self.training)
        return [pos, neg]


class PlausibilityDataset(Dataset):
    """Dataset for plausibility evaluation data."""
    
    def __init__(self, examples, preprocessor):
        self.examples = examples
        self.preprocessor = preprocessor

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, item):
        ex = self.examples[item]
        label, subj, verb, obj, subj_hyp, obj_hyp = ex
        
        return self.preprocessor.item_to_features(subj, verb, obj,
                                                  subj_hyp, obj_hyp, int(label), False)


class DataModule(pl.LightningDataModule):
    """Prepares data and dataloaders."""
    
    def __init__(
        self,
        model_type: str,
        model_name_or_path: str,
        cache_dir: str,
        data_dir: str,
        max_seq_length: int = 16,
        train_batch_size: int = 128,
        eval_batch_size: int = 128,
        num_workers: int = 10,
        **kwargs,
    ):
        super().__init__()
        self.model_type = model_type
        self.model_name_or_path = model_name_or_path
        self.cache_dir = cache_dir
        self.data_dir = data_dir
        self.max_seq_length = max_seq_length
        self.train_batch_size = train_batch_size
        self.eval_batch_size = eval_batch_size
        self.num_workers = num_workers

    def prepare_data(self):
        """ Parse, preprocess, and cache the datasets. """
        logging.info('Preparing/caching data')
        self.prepare_wikipedia_data()
        _ = utils.PreprocessingModule(self.data_dir, self.cache_dir,
                                      self.model_type, self.model_name_or_path)

    def setup(self, stage: str):
        logging.info('Setting up data module')
        # Preprocessing module
        preprocessor = utils.PreprocessingModule(self.data_dir, self.cache_dir,
                                                 self.model_type, self.model_name_or_path)
        self.dataset = {}
        # Wikipedia Data
        if stage == 'train':
            train_fname = os.path.join(self.cache_dir, 'train.pt')
            self.dataset['train'] = WikipediaDataset(torch.load(train_fname), preprocessor, True)
        val_fname   = os.path.join(self.cache_dir, 'val.pt')
        self.dataset['val'] = WikipediaDataset(torch.load(val_fname), preprocessor, False)
        # Plausibility data
        for dataset_name in ['pep_3k', 'twentyquestions']:
            for split in ['valid', 'test']:
                fname = os.path.join(self.data_dir, dataset_name, f'{split}.tsv')
                self.dataset[f'{dataset_name}_{split}'] = PlausibilityDataset(utils.read_tsv(fname),
                                                                              preprocessor)

    def train_dataloader(self):
        return DataLoader(
            self.dataset['train'],
            batch_size=self.train_batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=utils.collate_training_batch
        )

    def val_dataloader(self):
        return [
                    DataLoader(self.dataset['val'],
                               batch_size=self.eval_batch_size,
                               num_workers=self.num_workers,
                               collate_fn=utils.collate_training_batch),
                    DataLoader(self.dataset['pep_3k_valid'],
                               batch_size=self.eval_batch_size,
                               num_workers=self.num_workers,
                               collate_fn=utils.collate_batch),
                    DataLoader(self.dataset['twentyquestions_valid'],
                               num_workers=self.num_workers,
                               batch_size=self.eval_batch_size,
                               collate_fn=utils.collate_batch),
        ]
        
    def test_dataloader(self):
        return [
                    DataLoader(self.dataset['pep_3k_test'],
                               batch_size=self.eval_batch_size,
                               num_workers=self.num_workers,
                               collate_fn=utils.collate_batch),
                    DataLoader(self.dataset['twentyquestions_test'],
                               num_workers=self.num_workers,
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
