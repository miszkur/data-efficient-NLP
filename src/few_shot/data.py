import sys
sys.path.append( '..' )
sys.path.append( '.' )
from data_processing.ev_parser import create_dataset
import numpy as np
import random


class FewShotDataLoader:
    def __init__(self, num_classes=8, split='train'):
      self.ds = create_dataset(split=split)
      self.num_classes = num_classes
      self.create_datadict()

    def create_datadict(self):
      self.data_dict = {}
      for c in range(self.num_classes):
        self.data_dict[c] = {0: [], 1: []}

      for data_index in range(len(self.ds)):
        data_point = self.ds[data_index]
        label = data_point['label']
        for i, class_label in enumerate(label):
          self.data_dict[i][class_label.item()].append(data_point) 

    def create_episode(self, n_support: int = 0, n_classes: int = 0, n_query: int = 0, return_keys=False):
      episode = dict()
      n_classes = min(n_classes, len(self.data_dict.keys()))
      rand_keys = np.random.choice(list(self.data_dict.keys()), n_classes, replace=False)

      assert min([len(val[1]) for val in self.data_dict.values()]) >= n_support + n_query

      for key, val in self.data_dict.items():
          random.shuffle(val[0])
          random.shuffle(val[1])

      if n_support:
          episode["xs"] = [[self.data_dict[k][1][i] for i in range(n_support)] for k in rand_keys]
          episode["xn"] = [[self.data_dict[k][0][i] for i in range(n_support)] for k in rand_keys]
      if n_query:
          episode["xq"] = [[self.data_dict[k][1][n_support + i] for i in range(n_query)] for k in rand_keys]
          episode["label"] = [self.data_dict[k][1][n_support + i]['label'][rand_keys] for i in range(n_query) for k in rand_keys]
     
      if return_keys:
        return episode, rand_keys

      return episode

