import seaborn as sns
import matplotlib.pyplot as plt
import ml_collections
import pickle
import os

sns.set_style('darkgrid')

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
      plt.savefig(os.path.join(config.results_dir, f'accuracy.png'))

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
      plt.savefig(os.path.join(config.results_dir, f'f1_score.png'))

