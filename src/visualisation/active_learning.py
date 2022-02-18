import seaborn as sns
import matplotlib.pyplot as plt
import os

sns.set_style('darkgrid')

def plot_al_results(config):
  results_path = os.path.join(config.results_dir, f'{config.query_strategy}.pkl')
  
  with open(al_results_path, 'rb') as f:
    loaded_dict = pickle.load(f)

    # Plot accuracy.
    sns.lineplot(data=results, x="split", y="accuracy")
    plt.xlabel('Labeled data size')
    plt.ylabel('Accuracy')
    plt.savefig(os.path.join(config.results_dir, 'accuracy.png'))

    # Plot F1-score.
    sns.lineplot(data=results, x="split", y="f1_score")
    plt.xlabel('Labeled data size')
    plt.ylabel('Accuracy')
    plt.savefig(os.path.join(config.results_dir, 'f1_score.png'))

