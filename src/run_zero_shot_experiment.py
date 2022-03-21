from models.bart import BartEntailment
from data_processing.ev_parser import create_dataloader
from sklearn.metrics import classification_report
from tqdm import tqdm

import numpy as np
import ml_collections
import pickle
import os


def run_zero_shot_experiment(config: ml_collections.ConfigDict):

  model = BartEntailment(config)
  config.batch_size = 8
  (_, test_loader), class_names = create_dataloader(config, return_target_names=True, split='valid')
  class_name_to_idx = {}
  
  for i, class_name in enumerate(config.class_names):
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

  y_true = np.concatenate(y_true)
  y_pred = np.concatenate(y_preds) 
  results = classification_report(y_true, y_pred, target_names=config.class_names, zero_division=0, output_dict=True)
  correctly_classified = (y_pred == y_true)
  acc = correctly_classified.sum(axis=0) / y_pred.shape[0]
  for i, name in enumerate(config.class_names):
    results[name]['accuracy'] = acc[i]

  print('Saving results..')
  results_path = os.path.join(config.results_dir, f'test_results.pkl')
  with open(results_path, 'wb') as fp:
    pickle.dump(results, fp)

