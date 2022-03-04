import ml_collections
import numpy as np
import pickle
import torch
import os

from skmultilearn.model_selection import iterative_train_test_split
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


def initialize_strategy(strategy: Strategy, train_dataset, config, seed, al_class):
  if strategy == Strategy.RANDOM:
    return al.RandomStrategy(train_dataset, config.sample_size, config.batch_size, seed)
  elif strategy == Strategy.AVG_ENTROPY:
    return al.EntropyStrategy(
      train_dataset, strategy, config.sample_size, config.batch_size, seed)
  elif strategy == Strategy.MAX_ENTROPY:
    return al.EntropyStrategy(
      train_dataset, strategy, config.sample_size, config.batch_size, seed)
  elif strategy == Strategy.CLASS_ENTROPY:
    return al.EntropyStrategy(
      train_dataset, strategy, config.sample_size, config.batch_size, seed, class_index=al_class)
  elif strategy == Strategy.CAL:
    return al.CALStrategy(train_dataset, config.sample_size, config.batch_size, seed)

def get_stratified_sample(dataset, config, strategy):
  targets = dataset.targets

  X = np.arange(len(targets))
  y = np.array(targets)

  x_train, _, x_test, _ = iterative_train_test_split(
    X.reshape(-1, 1), y, test_size=config.sample_size / len(dataset))
  labeled_samples = torch.utils.data.Subset(dataset, x_test.reshape(-1,))
  unlabeled_samples = torch.utils.data.Subset(dataset, x_train.reshape(-1,))
  strategy.dataset = unlabeled_samples
  return labeled_samples

def run_active_learning_experiment(
  config: ml_collections.ConfigDict, 
  device: str, 
  strategy_type: Strategy,
  al_class: int,
  classes_to_track=[0,1],
  first_sample_stratified=False) -> Dict[str, List[float]]:
  """Run Active Learning experiment. 
  
  Save results to the folder specified in the config.
  For now only random strategy is used.

  Args:
      config (ml_collections): configuration dictionary.
      device (str): cpu or cuda. 
      strategy_type (Strategy): AL strategy to use.
      al_class (int): Class which will guide AL. (Not used by all strategies).
      classes_to_track (list, optional): Specifies for which classes metrics will be tracked. Defaults to [0,1] - 'functionality' and 'range_anxiety'.

  Returns:
      Dict[str, List[float]]: results dictionary with keys: 
      accuracy, f1_score and split (size of the training data) 
  """
  assert os.path.isdir(config.results_dir), 'Invalid path to save experiment results!'

  valid_loader, test_loader = create_dataloader(config, 'valid')
  train_dataset = create_dataset()
  num_al_iters = config.num_al_iters
  results = {'split': [], 'accuracy': [], 'f1_score':[]}
  for class_index in classes_to_track:
    results[class_index] = {
      'accuracy': [], 'f1_score':[], 'recall': [], 'precision':[]}

  for al_i, seed in enumerate(config.seeds):
    print(f'=== Active Learning experiment for seed {al_i+1}/{len(config.seeds)} ===')
    strategy = initialize_strategy(strategy_type, train_dataset, config, seed, al_class)

    if first_sample_stratified:
      labeled_data = get_stratified_sample(train_dataset, config, strategy)
    else:    
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
      learner.train(config, train_loader, valid_loader)
      # Test
      metrics = learner.evaluate(test_loader, classes=classes_to_track)
      loss = metrics['loss']
      accuracy = metrics['accuracy']
      f1_score = metrics['f1_score']
      print(f'Test loss: {loss}, accuracy: {accuracy}, f1 score: {f1_score}')
      results['accuracy'].append(accuracy)
      results['f1_score'].append(f1_score)
      for class_index in classes_to_track:
        for metric_name, value in metrics['classes'][class_index].items():
          if metric_name in list(results[class_index].keys()):
            results[class_index][metric_name].append(value)
      
      new_labeled_data = strategy.choose_samples_to_label(learner, train_loader)
      labeled_data = ConcatDataset([labeled_data, new_labeled_data])

    print('Saving results..')
    filename = config.query_strategy
    if first_sample_stratified:
      filename += '_stratified'
    results_path = os.path.join(config.results_dir, f'{filename}.pkl')
    with open(results_path, 'wb') as fp:
      pickle.dump(results, fp)

  return results
