import ml_collections
import torch
import os

from torch.utils.data import DataLoader, random_split, ConcatDataset
from abc import abstractmethod
from typing import Dict, List

from data_processing.ev_parser import create_dataset, create_dataloader
from visualisation.active_learning import plot_al_results
from config.config import multilabel_base
from models.bert import BertClassifier
from train import Learner


class QueryStrategy:
  """Base class for Active Learning query strategies."""
  def __init__(self, dataset, sample_size, seed):
    self.sample_size = sample_size
    self.generator = torch.Generator().manual_seed(seed)
    self.dataset = dataset

  @abstractmethod
  def choose_samples_to_label(self):
    pass

class RandomStrategy(QueryStrategy):
  """Chooses samples to label uniformly at random."""
  def __init__(self, dataset, sample_size=48, seed=42):
    super().__init__(dataset, sample_size, seed)

  def choose_samples_to_label(self):
    data_to_label, unlabeled_data = random_split(
      self.dataset, 
      [self.sample_size, len(self.dataset)-self.sample_size], 
      generator=self.generator)
    self.dataset = unlabeled_data
    return data_to_label


def run_active_learning_experiment(
  config: ml_collections.ConfigDict, device: str) -> Dict[str, List[float]]:
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

  for i, seed in enumerate(config.seeds):
    print(f'=== Active Learning experiment for seed {i+1}/{len(config.seeds)} ===')
    rqs = RandomStrategy(train_dataset, config.sample_size, seed)
    labeled_data = rqs.choose_samples_to_label()
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
      results['accuracy'].append(accuracy)
      results['f1_score'].append(f1_score)

      new_labeled_data = rqs.choose_samples_to_label()
      labeled_data = ConcatDataset([labeled_data, new_labeled_data])

  print('Saving results..')
  results_path = os.path.join(config.results_dir, f'{config.query_strategy}.pkl')
  with open(al_results_path, 'wb') as fp:
    pickle.dump(results, fp)
  plot_al_results(config)
  return results
