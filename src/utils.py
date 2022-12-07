import csv
import itertools
import os
import random

import numpy as np
import torch
from transformers import AutoTokenizer


def read_tsv(fname):
    with open(fname, 'r') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter='\t')
        return [row for row in csv_reader]


def read_tsv_as_dict(fname):
    data = {}
    for row in read_tsv(fname):
        data[row[0]] = row[1:]
    return data


def cache_df(df, fname):
    examples = df.to_numpy()
    torch.save(examples, fname)


def collate_batch(batch):
    input_ids = [x['input_ids'] for x in batch]
    attention_mask = [x['attention_mask'] for x in batch]
    labels = [x['labels'] for x in batch]

    output = {}
    output['input_ids'] = torch.vstack(input_ids)
    output['attention_mask'] = torch.vstack(attention_mask)
    output['labels'] = torch.concat(labels)

    if 'batch_marker' in batch[0]:
        batch_marker = [x['batch_marker'] for x in batch]
        output['batch_marker'] = torch.cumsum(torch.concat(batch_marker), dim=0) - 1

    return output


class WordNetData:
    def __init__(self, data_dir, cache_dir, model_type, model_name_or_path):
        self.data_dir = data_dir
        self.cache_dir = cache_dir
        self.model_type = model_type
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=False)
        self.load_dicts()
        
    def load_dicts(self):
        wordnet_cache_fname = os.path.join(self.cache_dir, 'wordnet.pt')
        if os.path.exists(wordnet_cache_fname):
            self.lemma2synsets, self.synset2lemma, self.synset2hc = torch.load(wordnet_cache_fname)
            
        lemma2synsets = read_tsv_as_dict(os.path.join(self.data_dir,
                                                      'wordnet',
                                                      'lemma2synsets.tsv'))
        synset2lemma = read_tsv_as_dict(os.path.join(self.data_dir,
                                                     'wordnet',
                                                     'synset2lemma.tsv'))
        synset2hc = read_tsv_as_dict(os.path.join(self.data_dir,
                                                  'wordnet',
                                                  'synset2hc.tsv'))
        
        torch.save((lemma2synsets, synset2lemma, synset2hc), wordnet_cache_fname)
        
    def item_to_features(self, *args):
        if self.model_type == 'roberta':
            return self.item_to_features_roberta(*args)
        elif self.model_type == 'conceptmax':
            return self.item_to_features_conceptmax(*args)
        else:
            raise ValueError(f'Invalid model type')
        
    def item_to_features_roberta(self, subj, verb, obj, subj_hyp, obj_hyp, label, training):
        texts = [' '.join([subj, verb, obj])]
        features = self.tokenizer(
            texts, max_length=16, padding='max_length', truncation=True,
            return_tensors='pt'
        )

        output = {}
        output['input_ids'] = features['input_ids'][0]
        output['attention_mask'] = features['attention_mask'][0]
        output['labels'] = torch.tensor([[label]], dtype=torch.float)[0]
        
        return output
    
    def item_to_features_conceptmax(self, subj, verb, obj, subj_hyp, obj_hyp, label, training):
        subj_hc = self.synset_to_hc(subj_hyp, subj)
        obj_hc = self.synset_to_hc(obj_hyp, obj)
        
        abstractions = list(itertools.product(subj_hc, obj_hc)) + \
            list(itertools.product([subj], obj_hc)) + \
            list(itertools.product(subj_hc, [obj]))
            
        if training:
            random.shuffle(abstractions)
            abstractions = abstractions[:3]
            
        abstractions += [(subj, obj)]
        
        texts = [f' {s} {verb} {o}' for s, o in abstractions]
        
        features = self.tokenizer(
            texts, max_length=16, padding='max_length', truncation=True,
            return_tensors='pt'
        )

        output = {}
        output['input_ids'] = features['input_ids']
        output['attention_mask'] = features['attention_mask']
        output['labels'] = torch.tensor([label], dtype=torch.float)
        output['batch_marker'] =  torch.zeros((len(texts)), dtype=torch.long)
        output['batch_marker'][0] = 1
        
        return output
    
    def synset_to_hc(self, synset, lemma):
        if not synset or synset == 'none' or synset not in self.synset2hc:
            return []
        synset_hc = self.synset2hc[synset]
        lemma_hc = [self.synset2lemma[x][0] for x in synset_hc if x != lemma]
        return list(set(lemma_hc))
