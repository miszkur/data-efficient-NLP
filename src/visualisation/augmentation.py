import os 
import pickle
import numpy as np

CLASS_NAMES = ['functionality','range_anxiety','availability','cost','ui','location','service_time','dealership']

def print_summary(results):
  for key in results.keys():
    if type(key) is int:
      print(f'Results for class {CLASS_NAMES[key]}')
      print_summary(results[key])
    else:
      print(f'{key}: {np.mean(results[key]):.2f} ({np.std(results[key]):.2f})')

def visualise_full_data_results(config):
  no_aug_results_path = os.path.join(config.results_dir, '..', 'supervised', f'SUPERVISED.pkl')
  results_path = os.path.join(config.results_dir, f'SUPERVISED.pkl')

  print('With data augmentation: ')
  with open(results_path, 'rb') as f:
    results = pickle.load(f)
    print_summary(results)
  
  print('\nWithout data augmentation: ')
  with open(no_aug_results_path, 'rb') as f:
    results = pickle.load(f)
    print_summary(results)

def visualise_small_data_results(config):
  no_aug_results_path = os.path.join(config.results_dir, 'small', f'SUPERVISED_300_aug_False.pkl')
  results_path = os.path.join(config.results_dir, 'small', f'SUPERVISED_300_aug_True.pkl')

  print('With data augmentation: ')
  with open(results_path, 'rb') as f:
    results = pickle.load(f)
    print_summary(results)
  
  print('\nWithout data augmentation: ')
  with open(no_aug_results_path, 'rb') as f:
    results = pickle.load(f)
    print_summary(results)

