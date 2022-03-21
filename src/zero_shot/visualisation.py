import matplotlib.pyplot as plt
import seaborn as sns

FIGURES_DIR = 'figures'

def visualise_per_class_performance(metric, config, filename='valid_results.pkl'):
  print(=====)
  results_path = os.path.join(config.results_dir, filename)
  with open(results_path, 'rb') as f:
    results = pickle.load(f)
    labels = config.class_names
    print(labels)
    metric_results = []
    for label in labels:
      metric_results.append(results[label][metric])

    plt.figure()
    sns.barplot(x=labels, y=metric_results)
    plt.savefig(os.path.join(config.results_dir, FIGURES_DIR, f'{metric}.png'))