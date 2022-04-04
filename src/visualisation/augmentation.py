import os 
import pickle
import numpy as np

class_names = ['functionality', 'range_anxiety']

def print_summary(results):
  keys = ['accuracy', 'f1_score', 'train_time', 0, 1, 'training_emissions']
  for key in results.keys():
    if key in [0,1]:
      print(f'Results for class {class_names[key]}')
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

