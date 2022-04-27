import torch
import numpy as np
import os
import pandas as pd

import sys
sys.path.append( '.' )
from data_processing.ev_parser import create_dataset

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, random_split, Subset, ConcatDataset
from abc import abstractmethod
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
from torch import nn
from enum import Enum
from tqdm import tqdm

class Strategy(Enum):
  RANDOM = 1
  MAX_ENTROPY = 2
  AVG_ENTROPY = 3
  CAL = 4

class QueryStrategy:
  """Base class for Active Learning query strategies."""
  def __init__(self, dataset, config, seed):
    self.sample_size = config.sample_size
    self.dataset = dataset
    self.generator = torch.Generator().manual_seed(seed)
    self.seed = seed
    self.batch_size = config.batch_size
    self.unlabeled_indexes = np.arange(start=0, stop=len(dataset))
    self.labeled_indexes = []
    self.use_aug_data = config.use_aug_data
    if self.use_aug_data:
      aug_data_path = os.path.join(config.data_dir, 'augmented_final.csv')
      self.df_aug = pd.read_csv(aug_data_path)
      train_data_path = os.path.join(config.data_dir, 'train_final.csv')
      self.df_train = pd.read_csv(train_data_path)

  @abstractmethod
  def choose_samples_to_label(self, learner, train_loader=None):
    pass

  def get_trainset_with_aug(self, indexes):
    augmented_sample = self.df_aug[np.isin(self.df_aug.id, indexes)]
    selected_sample = self.df_train[np.isin(self.df_train.id, indexes)]
    selected_sample = pd.concat([selected_sample, augmented_sample])
    selected_sample.drop(columns=['id'], inplace=True)
    selected_sample.reset_index(inplace=True)
    return create_dataset(df=selected_sample)

class RandomStrategy(QueryStrategy):
  """Chooses samples to label uniformly at random."""
  def __init__(self, dataset, config, seed=42):
    super().__init__(dataset, config, seed)

  def choose_samples_to_label(self, learner, train_loader=None):
    unlabeled_indexes, to_label_indexes = train_test_split(
      self.unlabeled_indexes, test_size=self.sample_size, random_state=self.seed)
    self.unlabeled_indexes = unlabeled_indexes
    self.labeled_indexes.append(to_label_indexes)
    if self.use_aug_data:
      return self.get_trainset_with_aug(to_label_indexes)
    
    return Subset(self.dataset, to_label_indexes)

class EntropyStrategy(QueryStrategy):
  """Chooses samples to label which have the maximum value of normalized entropy."""
  def __init__(self, dataset, strategy, config, seed=42):
    super().__init__(dataset, config, seed)
    if strategy == Strategy.AVG_ENTROPY:
      self.compute_entropy = self.avg_entropy
    elif strategy == Strategy.MAX_ENTROPY:
      self.compute_entropy = self.max_entropy

  def avg_entropy(self, probabilities):
    p = probabilities
    entropy = -(p*np.log(p)+(1-p)*np.log(1-p))
    return np.sum(entropy, axis=1) / entropy.shape[1]

  def max_entropy(self, probabilities):
    p = probabilities
    entropy = -(p*torch.log(p)+(1-p)*torch.log(1-p))
    return torch.max(entropy, axis=1).values

  def choose_samples_to_label(self, learner, train_loader=None):
    # Take random sample for the first iteration.
    if train_loader == None:
      unlabeled_indexes, to_label_indexes = train_test_split(
        self.unlabeled_indexes, test_size=self.sample_size, random_state=self.seed)
      self.unlabeled_indexes = unlabeled_indexes
      self.labeled_indexes.append(to_label_indexes)
      if self.use_aug_data:
        return self.get_trainset_with_aug(to_label_indexes)
      
      return Subset(self.dataset, to_label_indexes)

    unlabeled_dataset = Subset(self.dataset, self.unlabeled_indexes)
    data_loader = DataLoader(
      unlabeled_dataset, 
      batch_size=32,
      shuffle=False)

    entropies = []
    learner.model.eval()
    with torch.no_grad():
      for batch in tqdm(data_loader):
        output = learner.inference(batch) # batch_size x num_classes
        p = torch.sigmoid(output)
        entropies.append(self.compute_entropy(p))

    # Get the data to label based on entropy values.  
    entropies = torch.cat(entropies).cpu().numpy()

    partitioned_indices = np.argpartition(entropies, -self.sample_size)
    new_indices_to_label = partitioned_indices[-self.sample_size:]
    to_label_indexes = self.unlabeled_indexes[new_indices_to_label]
    self.unlabeled_indexes = np.delete(self.unlabeled_indexes, new_indices_to_label)
    self.labeled_indexes.append(to_label_indexes)
    if self.use_aug_data:
      return self.get_trainset_with_aug(to_label_indexes)
    
    return Subset(self.dataset, to_label_indexes)


class CALStrategy(QueryStrategy):
  """Chooses contranstive examples, see: https://arxiv.org/pdf/2109.03764.pdf."""
  def __init__(self, dataset, config, seed=42):
    super().__init__(dataset, config, seed)
    self.num_neighbors = 10 # default value used in CAL repo
    self.use_true_labels = False # True for ablation experiment.

  def process_labeled_data(self, learner, train_loader):
    learner.model.eval()
    train_logits_list = []
    train_emb_list = []
    train_labels_list = []
    with torch.no_grad():
      for batch in tqdm(train_loader):
        train_logits, train_emb = learner.inference(batch, return_cls=True)
        train_logits_list.append(train_logits)
        train_emb_list.append(train_emb)
        train_labels_list.append(batch['label'])

    train_logits = torch.cat(train_logits_list) 
    train_embeddings = torch.cat(train_emb_list) 
    train_labels = torch.cat(train_labels_list)
    return train_embeddings, train_logits, train_labels


  def choose_samples_to_label(self, learner, train_loader=None):
    # Take random sample for the first iteration.
    if train_loader == None:
      unlabeled_indexes, to_label_indexes = train_test_split(
        self.unlabeled_indexes, test_size=self.sample_size, random_state=self.seed)
      self.unlabeled_indexes = unlabeled_indexes
      self.labeled_indexes.append(to_label_indexes)
      if self.use_aug_data:
        return self.get_trainset_with_aug(to_label_indexes)
      
      return Subset(self.dataset, to_label_indexes)
    
    train_embeddings, train_logits, train_labels = self.process_labeled_data(learner, train_loader)
    neigh = KNeighborsClassifier(n_neighbors=self.num_neighbors) 
    neigh.fit(X=train_embeddings.cpu().numpy(), y=train_labels.numpy())
    criterion = nn.BCEWithLogitsLoss()
    num_classes = train_labels.shape[1]

    unlabeled_dataset = Subset(self.dataset, self.unlabeled_indexes)
    unlab_batch_size = 128
    data_loader = DataLoader(
      unlabeled_dataset, 
      batch_size=unlab_batch_size,
      shuffle=False)

    divergence_scores = []
    learner.model.eval()
    with torch.no_grad():
      for batch in tqdm(data_loader, desc="Finding neighbours for every unlabeled data point"):
        unlab_logits, unlab_embeddings = learner.inference(batch, return_cls=True) # batch_size x num_classes
        neighbour_indices = neigh.kneighbors(X=unlab_embeddings.cpu().numpy(),
        return_distance=False)

        batch_size = neighbour_indices.shape[0]

        if self.use_true_labels:
          # Ablation - use ground truth labels.
          neigh_labels = torch.reshape(
          train_labels[neighbour_indices.flatten()], 
          (batch_size, self.num_neighbors, num_classes)) 
        else:
          # "Original" CAL method - use model predicitons.
          neigh_logits = torch.reshape(
            train_logits[neighbour_indices.flatten()], 
            (batch_size, self.num_neighbors, num_classes))
          neigh_labels = torch.round(torch.sigmoid(neigh_logits))

        for i, label in enumerate(neigh_labels):
          x = unlab_logits[i].repeat(self.num_neighbors, 1)
          divergence = criterion(x, label)
          divergence_scores.append(divergence.cpu())
        
    partitioned_indices = np.argpartition(divergence_scores, -self.sample_size)
    new_indices_to_label = partitioned_indices[-self.sample_size:]
    to_label_indexes = self.unlabeled_indexes[new_indices_to_label]
    self.unlabeled_indexes = np.delete(self.unlabeled_indexes, new_indices_to_label)
    self.labeled_indexes.append(to_label_indexes)
    if self.use_aug_data:
      return self.get_trainset_with_aug(to_label_indexes)
    
    return Subset(self.dataset, to_label_indexes)
