import numpy as np
import random
import collections
import os
import json
import pandas as pd
from typing import List, Dict

DATA_DIR = '../../../../data'

def df_to_dict(df, shuffle=True):
    labels_dict = {
        0: list(df[df.functionality == 0].review),
        1: list(df[df.functionality == 1].review)
    }
    if shuffle:
        for key, val in labels_dict.items():
            random.shuffle(val)
    
    return labels_dict


class FewShotDataLoader:
    def __init__(self, split='train'):
        
        assert split in ['train', 'test', 'valid']

        self.df = pd.read_csv(os.path.join(DATA_DIR, f'{split}_final.csv'))
        
        shuffle = (split == 'train')
        self.data_dict = df_to_dict(self.df, shuffle=shuffle)

    def create_episode(self, n_support: int = 0, n_classes: int = 0, n_query: int = 0, n_unlabeled: int = 0, n_augment: int = 0):
        episode = dict()
        if n_classes:
            n_classes = min(n_classes, len(self.data_dict.keys()))
            rand_keys = np.random.choice(list(self.data_dict.keys()), n_classes, replace=False)

            assert min([len(val) for val in self.data_dict.values()]) >= n_support + n_query + n_unlabeled

            for key, val in self.data_dict.items():
                random.shuffle(val)

            if n_support:
                episode["xs"] = [[self.data_dict[k][i] for i in range(n_support)] for k in rand_keys]
            if n_query:
                episode["xq"] = [[self.data_dict[k][n_support + i] for i in range(n_query)] for k in rand_keys]

        return episode
