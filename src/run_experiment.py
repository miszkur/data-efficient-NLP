import ml_collections
import numpy as np
import pickle
import torch
import os

from torch.utils.data import DataLoader, random_split, ConcatDataset
from abc import abstractmethod
from typing import Dict, List
from enum import Enum
from tqdm import tqdm

from data_processing.ev_parser import create_dataset, create_dataloader
from visualisation.active_learning import plot_al_results
from config.config import multilabel_base
from models.bert import BertClassifier
from train import Learner

class Strategy(Enum):
  RANDOM = 1
  ENTROPY = 2

class QueryStrategy:
  """Base class for Active Learning query strategies."""
  def __init__(self, dataset, sample_size, batch_size):
    self.sample_size = sample_size
    self.dataset = dataset
    self.batch_size = batch_size

  @abstractmethod
  def choose_samples_to_label(self, learner: Learner):
    pass

class RandomStrategy(QueryStrategy):
  """Chooses samples to label uniformly at random."""
  def __init__(self, dataset, sample_size=48, batch_size=8, seed=42):
    super().__init__(dataset, sample_size, batch_size)
    self.generator = torch.Generator().manual_seed(seed)

  def choose_samples_to_label(self, learner: Learner):
    data_to_label, unlabeled_data = random_split(
      self.dataset, 
      [self.sample_size, len(self.dataset)-self.sample_size], 
      generator=self.generator)
    self.dataset = unlabeled_data
    return data_to_label

class MaxEntropyStrategy(QueryStrategy):
  """Chooses samples to label which have the maximum value of normalized entropy."""
  def __init__(self, dataset, sample_size=48, batch_size=8):
    super().__init__(dataset, sample_size, batch_size)

  def choose_samples_to_label(self, learner: Learner):
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

def initialize_strategy(strategy: Strategy, train_dataset, config, seed):
  if strategy == Strategy.RANDOM:
    return RandomStrategy(train_dataset, config.sample_size, config.batch_size, seed)
  elif strategy == Strategy.ENTROPY:
    return MaxEntropyStrategy(train_dataset, config.sample_size, config.batch_size)

def run_active_learning_experiment(
  config: ml_collections.ConfigDict, 
  device: str, 
  strategy_type: Strategy) -> Dict[str, List[float]]:
  """Run Active Learning experiment. 
  
  Save results to the folder specified in the config.
  For now only random strategy is used.

  Args:
      config (ml_collections): configuration dictionary.
      device (str): cpu or cuda. 

  Returns:
      Dict[str, List[float]]: results dictionary with keys: 
      accuracy, f1_score and split (size of the training data) 
  """
  assert os.path.isdir(config.results_dir), 'Invalid path to save experiment results!'

  test_loader = create_dataloader(config, 'test')
  train_dataset = create_dataset()
  num_al_iters = config.num_al_iters
  results = {'split': [], 'accuracy': [], 'f1_score':[]}

  for al_i, seed in enumerate(config.seeds):
    print(f'=== Active Learning experiment for seed {al_i+1}/{len(config.seeds)} ===')
    strategy = initialize_strategy(strategy_type, train_dataset, config, seed)
    
    model = BertClassifier(config=config.bert) 
    model.to(device)
    learner = Learner(device, model)

    labeled_data = strategy.choose_samples_to_label(learner)
    for i in range(num_al_iters):
      results['split'].append(len(labeled_data))
      train_loader = DataLoader(
        labeled_data, 
        batch_size=config.batch_size,
        shuffle=True)

      model = BertClassifier(config=config.bert) 
      model.to(device)
      # Train
      learner = Learner(device, model)
      learner.train(config, train_loader)
      # Test
      loss, accuracy, f1_score = learner.evaluate(test_loader)
      print(f'Test loss: {loss}, accuracy: {accuracy}, f1 score: {f1_score}')
      results['accuracy'].append(accuracy)
      results['f1_score'].append(f1_score)
      
      new_labeled_data = strategy.choose_samples_to_label(learner)
      labeled_data = ConcatDataset([labeled_data, new_labeled_data])

    print('Saving results..')
    results_path = os.path.join(config.results_dir, f'{config.query_strategy}.pkl')
    with open(results_path, 'wb') as fp:
      pickle.dump(results, fp)

  plot_al_results(config)
  return results
