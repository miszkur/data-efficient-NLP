import torch
import numpy as np

from sklearn.neighbors import KNeighborsClassifier
from torch.utils.data import DataLoader, random_split
from abc import abstractmethod
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
  def __init__(self, dataset, sample_size, batch_size):
    self.sample_size = sample_size
    self.dataset = dataset
    self.batch_size = batch_size

  @abstractmethod
  def choose_samples_to_label(self, learner, train_loader=None):
    pass

class RandomStrategy(QueryStrategy):
  """Chooses samples to label uniformly at random."""
  def __init__(self, dataset, sample_size=48, batch_size=8, seed=42):
    super().__init__(dataset, sample_size, batch_size)
    self.generator = torch.Generator().manual_seed(seed)

  def choose_samples_to_label(self, learner, train_loader=None):
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

  def choose_samples_to_label(self, learner, train_loader=None):
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

    partitioned_indices = np.argpartition(entropies, -self.sample_size)
    indices_to_label = partitioned_indices[-self.sample_size:]
    unlabeled_data_indices = partitioned_indices[:-self.sample_size]
    data_to_label = torch.utils.data.Subset(self.dataset, indices_to_label)
    self.dataset = torch.utils.data.Subset(self.dataset, unlabeled_data_indices)

    return data_to_label

class MaxEntropyStrategy(QueryStrategy):
  """Chooses samples to label which have the maximum value of normalized entropy."""
  def __init__(self, dataset, sample_size=48, batch_size=8):
    super().__init__(dataset, sample_size, batch_size)

  def choose_samples_to_label(self, learner, train_loader=None):
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

    entropies = np.concatenate(entropies)
    
    partitioned_indices = np.argpartition(entropies, -self.sample_size)
    indices_to_label = partitioned_indices[-self.sample_size:]
    unlabeled_data_indices = partitioned_indices[:-self.sample_size]
    data_to_label = torch.utils.data.Subset(self.dataset, indices_to_label)
    self.dataset = torch.utils.data.Subset(self.dataset, unlabeled_data_indices)

    return data_to_label


class CALStrategy(QueryStrategy):
  """Chooses contranstive examples, see: https://arxiv.org/pdf/2109.03764.pdf."""
  def __init__(self, dataset, sample_size=48, batch_size=8, seed=42):
    super().__init__(dataset, sample_size, batch_size)
    self.generator = torch.Generator().manual_seed(seed)
    self.num_neighbors = 10 # default value used in CAL repo

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
      data_to_label, unlabeled_data = random_split(
        self.dataset, 
        [self.sample_size, len(self.dataset)-self.sample_size], 
        generator=self.generator
      )
      self.dataset = unlabeled_data
      return data_to_label
    
    train_embeddings, train_logits, train_labels = self.process_labeled_data(learner, train_loader)
    neigh = KNeighborsClassifier(n_neighbors=self.num_neighbors) 
    neigh.fit(X=train_embeddings.cpu().numpy(), y=train_labels.numpy())
    criterion = nn.BCEWithLogitsLoss()
    num_classes = train_labels.shape[1]


    unlab_batch_size = 128
    data_loader = DataLoader(
      self.dataset, 
      batch_size=unlab_batch_size,
      shuffle=False)

    print(len(self.dataset))
  
    kl_scores = []
    num_adv = 0
    distances = []
    learner.model.eval()
    with torch.no_grad():
      for batch in tqdm(data_loader, desc="Finding neighbours for every unlabeled data point"):
        # if num_adv < 66:
        #   num_adv += 1
        #   continue
        # print('e')
        unlab_logits, unlab_embeddings = learner.inference(batch, return_cls=True) # batch_size x num_classes
        neighbour_indices = neigh.kneighbors(X=unlab_embeddings.cpu().numpy(),
        return_distance=False)
        # Labeled neigh labels
        batch_size = neighbour_indices.shape[0]
        neigh_labels = torch.reshape(
          train_labels[neighbour_indices.flatten()], 
          (batch_size, self.num_neighbors, num_classes)) 
        neigh_logits = torch.reshape(
          train_logits[neighbour_indices.flatten()], 
          (batch_size, self.num_neighbors, num_classes)) 
        neigh_preds = torch.round(torch.sigmoid(neigh_logits))

        unlab_pred = torch.round(torch.sigmoid(unlab_logits))
        for i, label in enumerate(neigh_labels):
          x = unlab_logits[i].repeat(self.num_neighbors, 1)
          kl = criterion(x.cpu(), label)
          kl_scores.append(kl)
        
    partitioned_indices = np.argpartition(kl_scores, -self.sample_size)
    indices_to_label = partitioned_indices[-self.sample_size:]
    unlabeled_data_indices = partitioned_indices[:-self.sample_size]
    data_to_label = torch.utils.data.Subset(self.dataset, indices_to_label)
    self.dataset = torch.utils.data.Subset(self.dataset, unlabeled_data_indices)
    return data_to_label
