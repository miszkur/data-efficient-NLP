import seaborn as sns
import matplotlib.pyplot as plt
import ml_collections
import pickle
import numpy as np
import os

sns.set_style('darkgrid')
FIGURES_DIR = 'figures'

def plot_al_results(strategies, config: ml_collections.ConfigDict):
  """Plot accuracy and F1 score of AL experiment

  Args:
      config (ml_collections.ConfigDict): configuration dictionary. 
      It should contain path to experiment results and query strategy used in AL.
  """
  for strategy in strategies:
    results_path = os.path.join(config.results_dir, f'{strategy}.pkl')
    
    with open(results_path, 'rb') as f:
      results = pickle.load(f)

      # Plot accuracy.
      sns.lineplot(data=results, x="split", y="accuracy", label=strategy)
      plt.legend()
      plt.xlabel('Labeled data size')
      plt.ylabel('Accuracy')
      plt.title('Accuracy for different data sizes')
      x_axis = results['split']

  results_path = os.path.join(config.results_dir, f'SUPERVISED.pkl')
  with open(results_path, 'rb') as f:
    results_supervised = pickle.load(f)
    supervised_results_dict = {'split': [], 'accuracy': [], 'f1_score':[]}
    for split in set(x_axis):
      for k, v in results_supervised.items():
        supervised_results_dict[k] += v  
      supervised_results_dict['split'] += [split for _ in range(len(v))]

    sns.lineplot(data=supervised_results_dict, x='split', y='accuracy', label='Full supervision', color='black')
  plt.savefig(os.path.join(config.results_dir, FIGURES_DIR, f'accuracy.png'))

  # Plot F1-score.
  plt.figure()
  for strategy in strategies:
    results_path = os.path.join(config.results_dir, f'{strategy}.pkl')
    
    with open(results_path, 'rb') as f:
      results = pickle.load(f)
      sns.lineplot(data=results, x="split", y="f1_score", label=strategy)
      plt.legend()
      plt.xlabel('Labeled data size')
      plt.ylabel('F1 score')
      plt.title('F1 score for different data sizes')

  sns.lineplot(data=supervised_results_dict, x='split', y='f1_score', label='Full supervision', color='black')
  plt.savefig(os.path.join(config.results_dir, FIGURES_DIR, f'f1_score.png'))

def cold_vs_warm_start(strategy: str, config):
  results_path = os.path.join(config.results_dir, f'{strategy}.pkl')
  results_path_cold_start = os.path.join(config.results_dir, f'{strategy}_cold_start.pkl')
  

  with open(results_path, 'rb') as f:
    results = pickle.load(f)
    with open(results_path_cold_start, 'rb') as f:
      results_cold_start = pickle.load(f)

      # Plot accuracy.
      plt.figure()
      sns.lineplot(data=results, x="split", y="accuracy", label=strategy)
      sns.lineplot(data=results_cold_start, x="split", y="accuracy", label=f'{strategy} cold start')
      plt.legend()
      plt.xlabel('Labeled data size')
      plt.ylabel('Accuracy')
      plt.title('Accuracy for different data sizes')
      plt.savefig(os.path.join(config.results_dir, FIGURES_DIR, f'{strategy}_accuracy.png'))

      plt.figure()
      sns.lineplot(data=results, x="split", y="f1_score", label=strategy)
      sns.lineplot(data=results_cold_start, x="split", y="f1_score", label=f'{strategy} cold start')
      plt.legend()
      plt.xlabel('Labeled data size')
      plt.ylabel('F1 score')
      plt.title('F1 score for different data sizes')
      plt.savefig(os.path.join(config.results_dir, FIGURES_DIR, f'{strategy}_f1_score.png'))

def plot_metrics_for_classes(config, metrics, classes, strategies, class_names, savedir='class_results'):
  for class_index in classes:
    for metric in metrics:
      plt.figure()
      for strategy in strategies:
        results_path = os.path.join('../results/al', f'{strategy}.pkl')
        
        with open(results_path, 'rb') as f:
          results = pickle.load(f)

          sns.lineplot(x=results['split'], y=results[class_index][metric], label=strategy)
          plt.legend()
          plt.xlabel('Labeled data size')
          plt.ylabel(metric)
          plt.title(f'{metric} for different data sizes for {class_names[class_index]} class')
      plt.savefig(os.path.join(config.results_dir, FIGURES_DIR, savedir, f'{metric}_class{class_index}.png'))