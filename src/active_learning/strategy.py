import torch
import numpy as np

from torch.utils.data import DataLoader, random_split
from abc import abstractmethod
from enum import Enum
from tqdm import tqdm

class Strategy(Enum):
  RANDOM = 1
  MAX_ENTROPY = 2
  AVG_ENTROPY = 3

class QueryStrategy:
  """Base class for Active Learning query strategies."""
  def __init__(self, dataset, sample_size, batch_size):
    self.sample_size = sample_size
    self.dataset = dataset
    self.batch_size = batch_size

  @abstractmethod
  def choose_samples_to_label(self, learner):
    pass

class RandomStrategy(QueryStrategy):
  """Chooses samples to label uniformly at random."""
  def __init__(self, dataset, sample_size=48, batch_size=8, seed=42):
    super().__init__(dataset, sample_size, batch_size)
    self.generator = torch.Generator().manual_seed(seed)

  def choose_samples_to_label(self, learner):
    data_to_label, unlabeled_data = random_split(
      self.dataset, 
      [self.sample_size, len(self.dataset)-self.sample_size], 
      generator=self.generator)
    self.dataset = unlabeled_data
    return data_to_label

class AvgEntropyStrategy(QueryStrategy):
  """Chooses samples to label which have the maximum value of normalized entropy."""
  def __init__(self, dataset, sample_size=48, batch_size=8):
    super().__init__(dataset, sample_size, batch_size)

  def choose_samples_to_label(self, learner):
    data_loader = DataLoader(
      self.dataset, 
      batch_size=self.batch_size,
      shuffle=False)

    entropies = []
    learner.model.eval()
    with torch.no_grad():
      for batch in tqdm(data_loader):
        output = learner.inference(batch) # batch_size x num_classes
        p = torch.sigmoid(output).cpu().detach().numpy()
        entropy = -(p*np.log(p)+(1-p)*np.log(1-p))
        normalized_entropy = np.sum(entropy, axis=1) / entropy.shape[1]
        entropies.append(normalized_entropy)

    # Get the data to label based on entropy values.  
    entropies = np.concatenate(entropies)
    sorted_indices = np.flip(np.argsort(entropies))
    indices_to_label = sorted_indices[:self.sample_size]
    data_to_label = torch.utils.data.Subset(self.dataset, indices_to_label)

    # Get the rest of dataset (unlabeled).
    all_indices = np.arange(len(self.dataset))
    unlabeled_data_indices = all_indices[~np.in1d(all_indices, indices_to_label)]
    self.dataset = torch.utils.data.Subset(self.dataset, unlabeled_data_indices)
    return data_to_label

class MaxEntropyStrategy(QueryStrategy):
  """Chooses samples to label which have the maximum value of normalized entropy."""
  def __init__(self, dataset, sample_size=48, batch_size=8):
    super().__init__(dataset, sample_size, batch_size)

  def choose_samples_to_label(self, learner):
    data_loader = DataLoader(
      self.dataset, 
      batch_size=self.batch_size,
      shuffle=False)

    entropies = []
    learner.model.eval()
    with torch.no_grad():
      for batch in tqdm(data_loader):
        output = learner.inference(batch) # batch_size x num_classes
        p = torch.sigmoid(output).cpu().detach().numpy()
        entropy = -(p*np.log(p)+(1-p)*np.log(1-p))
        max_entropy = np.max(entropy, axis=1)
        entropies.append(max_entropy)

    # Get the data to label based on entropy values.  
    entropies = np.concatenate(entropies)
    sorted_indices = np.flip(np.argsort(entropies))
    indices_to_label = sorted_indices[:self.sample_size]
    data_to_label = torch.utils.data.Subset(self.dataset, indices_to_label)

    # Get the rest of dataset (unlabeled).
    all_indices = np.arange(len(self.dataset))
    unlabeled_data_indices = all_indices[~np.in1d(all_indices, indices_to_label)]
    self.dataset = torch.utils.data.Subset(self.dataset, unlabeled_data_indices)
    return data_to_label