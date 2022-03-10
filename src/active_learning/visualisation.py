import seaborn as sns
import matplotlib.pyplot as plt
import ml_collections
import pickle
import numpy as np
import os

sns.set_style('darkgrid')
FIGURES_DIR = 'figures'

def plot_al_results(strategies, config: ml_collections.ConfigDict, metrics):
  """Plot accuracy and F1 score of AL experiment

  Args:
      config (ml_collections.ConfigDict): configuration dictionary. 
      It should contain path to experiment results and query strategy used in AL.
  """
  results_path = os.path.join(config.results_dir, f'SUPERVISED.pkl')
  with open(results_path, 'rb') as f:
    results_supervised = pickle.load(f)

    for metric in metrics:
      plt.figure()
      for strategy in strategies:
        results_path = os.path.join(config.results_dir, f'{strategy}.pkl')
        
        with open(results_path, 'rb') as f:
          results = pickle.load(f)
          y = results[metric]
          # Plot accuracy.
          sns.lineplot(x=results["split"][-len(y):], y=y, label=strategy)
          
          plt.xlabel('Labeled data size')
          plt.ylabel(metric)
          plt.title(f'{metric} for different data sizes')

      
        if metric in results_supervised:
          plt.axhline(y = np.mean(results_supervised[metric]), color = 'r', linestyle = '--', label='Full supervision', c='black')
      plt.legend()
      plt.savefig(os.path.join(config.results_dir, FIGURES_DIR, f'{metric}.png'))


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
  results_path = os.path.join(config.results_dir, f'SUPERVISED.pkl')
  with open(results_path, 'rb') as f:
    results_supervised = pickle.load(f)
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
      
        if class_index in results_supervised and metric in results_supervised[class_index]:
          plt.axhline(
            y = np.mean(results_supervised[class_index][metric]), 
            color = 'r', linestyle = '--', label='Full supervision', c='black')
      
        plt.savefig(os.path.join(config.results_dir, FIGURES_DIR, savedir, f'{metric}_class{class_index}.png'))
