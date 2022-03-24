import matplotlib.pyplot as plt
import seaborn as sns
import os
import pickle

FIGURES_DIR = 'figures'


def visualise_per_class_performance(
  config, 
  metrics=['precision', 'recall', 'accuracy'],
  filename='valid_results.pkl'):
  sns.set_palette(sns.color_palette("Set2"))
  results_path = os.path.join(config.results_dir, filename)
  with open(results_path, 'rb') as f:
    results = pickle.load(f)
    print(results)
    labels = config.class_names
    fig, axes = plt.subplots(3,1, figsize=(6, 10))
    for metric, ax in zip(metrics, axes.flatten()):
      metric_results = []
      for label in labels:
        # label = label.replace(' ', '_') # range anxiety => range_anxiety.
        metric_results.append(results[label][metric])

      # plt.figure()
      ax.set_title(f'{metric}')
      # plt.xlabel(metric)
      ax = sns.barplot(x=labels, y=metric_results, ax=ax)
      ax.set_xticklabels([])
      ax.set_ylim([0,1])

    clicks_no_preannotation = 0
    clicks_preannotation = 0
    print('\nClass name & Clicks pre-annotation & Clicks without pre-annotation \\\\ \hline')

    for label in labels:
      # 533 is the data size.
      new_clicks_preannotation = 533 - results[label]['accuracy']*533
      new_clicks_no_preannotation = results[label]['support']
      print(f'{label} & {new_clicks_preannotation} & {new_clicks_no_preannotation} \\\\')
      clicks_no_preannotation += new_clicks_no_preannotation
      clicks_preannotation += new_clicks_preannotation

    print('\hline\hline')
    print(f'sum & {clicks_preannotation} & {clicks_no_preannotation}')

    ax.set_xticklabels(labels)
    ax.tick_params(axis='x', rotation=90)
    plt.suptitle('Zero-shot results')
    plt.savefig(os.path.join(config.results_dir, FIGURES_DIR, f'all_metrics.png'))
