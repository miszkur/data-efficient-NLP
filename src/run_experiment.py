import ml_collections
import numpy as np
import pickle
import torch
import os

from torch.utils.data import ConcatDataset, DataLoader
from typing import Dict, List
from tqdm import tqdm

from data_processing.ev_parser import create_dataset, create_dataloader
from config.config import multilabel_base
from models.bert import BertClassifier
from learner import Learner

from active_learning.strategy import Strategy
import active_learning.strategy as al
from active_learning.visualisation import plot_al_results


def initialize_strategy(strategy: Strategy, train_dataset, config, seed):
  if strategy == Strategy.RANDOM:
    return al.RandomStrategy(train_dataset, config.sample_size, config.batch_size, seed)
  elif strategy == Strategy.AVG_ENTROPY:
    return al.AvgEntropyStrategy(train_dataset, config.sample_size, config.batch_size, seed)
  elif strategy == Strategy.MAX_ENTROPY:
    return al.MaxEntropyStrategy(train_dataset, config.sample_size, config.batch_size, seed)
  elif strategy == Strategy.CAL:
    return al.CALStrategy(train_dataset, config.sample_size, config.batch_size, seed)

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
      
      new_labeled_data = strategy.choose_samples_to_label(learner, train_loader)
      labeled_data = ConcatDataset([labeled_data, new_labeled_data])

    print('Saving results..')
    results_path = os.path.join(config.results_dir, f'{config.query_strategy}.pkl')
    with open(results_path, 'wb') as fp:
      pickle.dump(results, fp)

  return results
