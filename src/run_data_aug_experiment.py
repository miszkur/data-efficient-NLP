import numpy as np
import pandas as pd
import ml_collections
import time
import pickle
import os
import torch

from codecarbon import EmissionsTracker
from torch.utils.data import ConcatDataset, Subset, DataLoader

from data_processing.ev_parser import create_dataloader, create_dataset
from models.bert import BertClassifier
from learner import Learner


def initialize_results_dict(classes_to_track):
  results = {
    'accuracy': [], 
    'f1_score':[], 
    'train_time': [], 
    'training_emissions': [], 
    'fp_fn': []
  }
  for class_index in classes_to_track:
    results[class_index] = {
      'accuracy': [], 'f1_score':[], 'recall': [], 'precision':[]}
  return results

def train_all_data(
  config: ml_collections.ConfigDict, 
  device: str,
  classes_to_track=[0,1]):

  valid_loader, test_loader = create_dataloader(config, 'valid')
  results = initialize_results_dict(classes_to_track)

  train_dataset = create_dataset()
  augmented_dataset = create_dataset(split='augmented')
  merged_ds = ConcatDataset([train_dataset, augmented_dataset])

  for _ in range(3):
    train_loader = DataLoader(
      merged_ds, 
      batch_size=config.batch_size,
      shuffle=True)

    model = BertClassifier(config=config.bert) 
    model.to(device)
    # Train
    learner = Learner(device, model, config.results_dir)
    tracker = EmissionsTracker()
    tracker.start()
    train_start_time = time.time()
    
    learner.train(
      config,
      train_loader=train_loader,
      validation_loader=valid_loader)

    results['train_time'].append(time.time() - train_start_time)
    results['training_emissions'].append(tracker.stop())
    
    # Evaluate
    metrics = learner.evaluate(test_loader, classes=classes_to_track)
    loss = metrics['loss']
    accuracy = metrics['accuracy']
    f1_score = metrics['f1_score']
    print(f'Test loss: {loss}, accuracy: {accuracy}, f1 score: {f1_score}')
    results['accuracy'].append(accuracy)
    results['f1_score'].append(f1_score)
    results['fp_fn'].append(metrics['fp_fn'])
    for class_index in classes_to_track:
      for metric_name, value in metrics['classes'][class_index].items():
        if metric_name in list(results[class_index].keys()):
          results[class_index][metric_name].append(value)

    print('Saving results..')
    results_path = os.path.join(config.results_dir, 'full', f'SUPERVISED.pkl')
    with open(results_path, 'wb') as fp:
      pickle.dump(results, fp)



def train_limited_data(
  config: ml_collections.ConfigDict, 
  device: str,
  with_augmentations=True,
  data_size=300,
  classes_to_track=[0,1,2,3,4,5,6,7]):

  valid_loader, test_loader = create_dataloader(config, 'valid')
  results = initialize_results_dict(classes_to_track)

  # train_dataset = create_dataset()
  # augmented_dataset = create_dataset(split='augmented')

  train_data_path = os.path.join(config.data_dir, 'train_final.csv')
  aug_data_path = os.path.join(config.data_dir, 'augmented_final.csv')
  df_train = pd.read_csv(train_data_path)
  df_aug = pd.read_csv(aug_data_path)

  for i in range(5):
    selected_sample = df_train.sample(n=data_size, random_state=config.seeds[i])
    if with_augmentations:
      augmented_sample = df_aug[np.isin(df_aug.id, selected_sample.id)]
      selected_sample = pd.concat([selected_sample, augmented_sample])
    selected_sample.drop(columns=['id'], inplace=True)
    selected_sample.reset_index(inplace=True)
    train_dataset = create_dataset(df=selected_sample)

    train_loader = torch.utils.data.DataLoader(
      train_dataset, 
      batch_size=config.batch_size,
      shuffle=True)

    model = BertClassifier(config=config.bert) 
    model.to(device)
    # Train
    learner = Learner(device, model, config.results_dir)
    tracker = EmissionsTracker()
    tracker.start()
    train_start_time = time.time()
    
    learner.train(
      config,
      train_loader=train_loader,
      validation_loader=valid_loader)

    results['train_time'].append(time.time() - train_start_time)
    results['training_emissions'].append(tracker.stop())
    
    # Evaluate
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

    print('Saving results..')
    results_path = os.path.join(
      config.results_dir, 'small', f'SUPERVISED_{data_size}_aug_{with_augmentations}.pkl')
    with open(results_path, 'wb') as fp:
      pickle.dump(results, fp)
