from data_processing.ev_parser import create_dataset, create_dataloader
from config.config import multilabel_base
from models.bert import BertClassifier
from torch.utils.data import DataLoader
from torch.utils.data import random_split, ConcatDataset
from abc import abstractmethod
from train import Learner

import torch
import os

class QueryStrategy:
  def __init__(self, dataset, sample_size, seed):
    self.sample_size = sample_size
    self.generator = torch.Generator().manual_seed(seed)
    self.dataset = dataset

  @abstractmethod
  def choose_samples_to_label(self):
    pass

class RandomStrategy(QueryStrategy):
  def __init__(self, dataset, sample_size=48, seed=42):
    super().__init__(dataset, sample_size, seed)

  def choose_samples_to_label(self):
    data_to_label, unlabeled_data = random_split(
      self.dataset, 
      [self.sample_size, len(self.dataset)-self.sample_size], 
      generator=self.generator)
    self.dataset = unlabeled_data
    return data_to_label


def run_active_learning_experiment(config, device):
  """
  Run Active Learning experiment. For now only random strategy is used.

  Args:
      config: The configuration dictionary.

  Returns:
      The experiment result.
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
  return results
