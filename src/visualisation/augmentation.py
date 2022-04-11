import os 
import pickle
import numpy as np

CLASS_NAMES = ['functionality','range_anxiety','availability','cost','ui','location','service_time','dealership']
class_metrics = ['accuracy', 'f1_score', 'incorrect_predictions']
general_metrics = ['accuracy', 'f1_score', 'train_time']
per_class_support = {
  'functionality': 286,
  'range_anxiety': 23,
  'availability': 101,
  'cost': 50,
  'ui': 65,
  'location': 183,
  'service_time': 49,
  'dealership': 75
}


def print_summary(results, results_no_aug):
  for key in results.keys():
    if type(key) is int:
      print(f'Results for class {CLASS_NAMES[key]}')
      print_summary(results[key], results_no_aug[key])
    else:
      num_spaces_left = 25 - len(key)
      print(f'{key} {num_spaces_left*" "} | {np.mean(results[key]):.2f} ({np.std(results[key]):.2f}) \
      | {np.mean(results_no_aug[key]):.2f} ({np.std(results_no_aug[key]):.2f})')


def print_latex_summary(results, results_no_aug):
  incorrect_predictions_sum = np.zeros(5)
  incorrect_predictions_no_aug_sum = np.zeros(5)
  for key in results.keys():
    if type(key) is int:
      incorrect_predictions_sum += np.array(results[key]['incorrect_predictions'])
      incorrect_predictions_no_aug_sum += np.array(results_no_aug[key]['incorrect_predictions'])

      print(f'\n{CLASS_NAMES[key]} & No ', end='')
      for metric in class_metrics:
        print(f' & {np.mean(results_no_aug[key][metric]):.2f} ({np.std(results_no_aug[key][metric]):.2f}) ', end='')
      
      print(f' & {per_class_support[CLASS_NAMES[key]]}', end='')
      print('\\\\')
      print(f' & Yes', end='')
      for metric in class_metrics:
        print(f' & {np.mean(results[key][metric]):.2f} ({np.std(results[key][metric]):.2f}) ', end='')
      print(' & \\\\ \\hline')

  # Print the header.
  print('\nAugmentation ', end='')
  for metric in general_metrics:
    print(f' & {metric}', end='')
  print(' & ENUA \\\\ \hline')

  print('No', end='')
  for metric in general_metrics:
    print(f' & {np.mean(results_no_aug[metric]):.2f} ({np.std(results_no_aug[metric]):.2f}) ', end='')
  print(f' & {np.mean(incorrect_predictions_no_aug_sum):.2f} ({np.std(incorrect_predictions_no_aug_sum):.2f}) \\\\')
  print('Yes', end='')
  for metric in general_metrics:
    print(f' & {np.mean(results[metric]):.2f} ({np.std(results[metric]):.2f}) ', end='')
  print(f' & {np.mean(incorrect_predictions_sum):.2f} ({np.std(incorrect_predictions_sum):.2f}) \\\\')


def visualise_augmentation_results(config, aug_mode='small', data_size=300):
  if aug_mode == 'full':
    filename = 'SUPERVISED'
  else:
    filename = f'SUPERVISED_{data_size}'

  no_aug_results_path = os.path.join(config.results_dir, aug_mode, f'{filename}_aug_False.pkl')
  results_path = os.path.join(config.results_dir, aug_mode, f'{filename}_aug_True.pkl')

  with open(results_path, 'rb') as f:
    results = pickle.load(f)
    with open(no_aug_results_path, 'rb') as f2:
      results_no_aug = pickle.load(f2)
      print_latex_summary(results, results_no_aug)
  
