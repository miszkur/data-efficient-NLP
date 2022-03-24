from models.bart import BartEntailment
from data_processing.ev_parser import create_dataloader, create_dataset
from sklearn.metrics import classification_report
from tqdm import tqdm

import gensim.downloader
import numpy as np
import ml_collections
import pickle
import os


def run_zero_shot_experiment(
  config: ml_collections.ConfigDict, 
  class_names, 
  results_filename='test_results',
  class_index=None):
  config.class_names = class_names
  model = BartEntailment(config)
  config.batch_size = 8
  (_, test_loader) = create_dataloader(config, split='valid')
  class_name_to_idx = {}
  
  for i, class_name in enumerate(class_names):
    class_name_to_idx[class_name] = i

  num_classes = len(class_names)

  y_true = []
  y_preds = []
  for x in tqdm(test_loader):
    predictions = model.predict(x['review_text'])
    y_pred = np.zeros((len(predictions),num_classes))
    for prediction_id, prediction in enumerate(predictions):
      for i, score in enumerate(prediction['scores']):
        if score < 0.5:
          # Scores are sorted in descending order.
          break

        label = prediction['labels'][i]
        y_pred[prediction_id, class_name_to_idx[label]] = 1
    y_preds.append(y_pred)
    y_true.append(x['label'].cpu().numpy())

  if class_index is not None:
    y_pred = np.concatenate(y_preds) 
    y_true = np.concatenate(y_true)
    y_true = np.tile(y_true[:, class_index].reshape(-1, 1), (1, num_classes))
  else:
    y_true = np.concatenate(y_true)
    y_pred = np.concatenate(y_preds) 
    
  results = classification_report(y_true, y_pred, target_names=class_names, zero_division=0, output_dict=True)
  correctly_classified = (y_pred == y_true)
  acc = correctly_classified.sum(axis=0) / y_pred.shape[0]
  for i, name in enumerate(class_names):
    results[name]['accuracy'] = acc[i]
    results[name]['num_positive_predictions'] = np.sum(y_pred[:,i])

  print('Saving results..')
  results_path = os.path.join(config.results_dir, f'{results_filename}.pkl')
  with open(results_path, 'wb') as fp:
    pickle.dump(results, fp)


def run_zero_shot_for_semantic_neighbors(config, embeddings='glove-wiki-gigaword-50'):

  semantic_neighbors_path = os.path.join(config.results_dir, f'{embeddings}.pkl')

  if not os.path.exists(semantic_neighbors_path):
    class_names = config.class_names
    neighbors_per_class = 9
    glove_vectors = gensim.downloader.load(embeddings)  

    class_index_to_neighbors = {}
    for class_i, class_name in enumerate(class_names):
      if ' ' in class_name or 'ui' in class_name:
        continue

      class_index_to_neighbors[class_i] = []
      semantic_neighbors = glove_vectors.most_similar(class_name, topn=100)
      # Filter out words not present in the corpus.
      with open('../data/train_final.csv') as file:
        contents = file.read()
        neighbors_in_corpus = list(
          filter(lambda neighbor: neighbor[0] in contents, semantic_neighbors)
        )
      class_index_to_neighbors[class_i].append(class_name)
      for i in range(neighbors_per_class):
        class_index_to_neighbors[class_i].append(neighbors_in_corpus[i][0])

    with open(semantic_neighbors_path, 'wb') as fp:
      pickle.dump(class_index_to_neighbors, fp)
  else:
    with open(semantic_neighbors_path, 'rb') as fp:
      class_index_to_neighbors = pickle.load(fp)

  print(class_index_to_neighbors)
      
  for class_i, neighbors in class_index_to_neighbors.items():
    print(f'Running zero-shot for classes: {neighbors}...')
    run_zero_shot_experiment(config, neighbors, f'{neighbors[0]}_neighbors', class_index=class_i)