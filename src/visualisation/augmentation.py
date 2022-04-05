import os 
import pickle
import numpy as np

CLASS_NAMES = ['functionality','range_anxiety','availability','cost','ui','location','service_time','dealership']

def print_summary(results, results2):
  for key in results.keys():
    if type(key) is int:
      print(f'Results for class {CLASS_NAMES[key]}')
      print_summary(results[key], results2[key])
    else:
      num_spaces_left = 25 - len(key)
      print(f'{key} {num_spaces_left*" "} | {np.mean(results[key]):.2f} ({np.std(results[key]):.2f}) \
      | {np.mean(results2[key]):.2f} ({np.std(results2[key]):.2f})')

def visualise_full_data_results(config):
  no_aug_results_path = os.path.join(config.results_dir, '..', 'supervised', f'SUPERVISED.pkl')
  results_path = os.path.join(config.results_dir, 'full', f'SUPERVISED.pkl')

  with open(results_path, 'rb') as f:
    results = pickle.load(f)
    with open(no_aug_results_path, 'rb') as f2:
      results_no_aug = pickle.load(f2)
      print_summary(results, results_no_aug)

def visualise_small_data_results(config):
  no_aug_results_path = os.path.join(config.results_dir, 'small', f'SUPERVISED_300_aug_False.pkl')
  results_path = os.path.join(config.results_dir, 'small', f'SUPERVISED_300_aug_True.pkl')

  with open(results_path, 'rb') as f:
    results = pickle.load(f)
    with open(no_aug_results_path, 'rb') as f2:
      results_no_aug = pickle.load(f2)
      print_summary(results, results_no_aug)
  
