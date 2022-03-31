"""Script for backtranslating the dataset."""

import nlpaug.augmenter.word as naw
import config.config as configs
import os
import csv

from data_processing.ev_parser import create_dataset, create_dataloader
from tqdm import tqdm


results_path = os.path.join('..', 'results', 'backtranslation', 'augmented.csv')
CLASS_NAMES = ['functionality','range_anxiety','availability','cost','ui','location','service_time','dealership']
HEADERS = ['id','original','augmented','functionality','range_anxiety','availability','cost','ui','location','service_time','dealership']


aug = naw.BackTranslationAug()

config = configs.multilabel_base()
config.batch_size = 2
train_loader = create_dataloader(config, is_training=False)

results = []

index = 0
for batch in tqdm(train_loader):
  labels = batch['label'].numpy()
  for text_id, text in enumerate(tqdm(batch['review_text'])):
    line = {
      'id': index,
      'original': text,
      'augmented': aug.augment(text)
    }

    for i, y_true in enumerate(labels[text_id]):
      line[CLASS_NAMES[i]] = int(y_true)
    results.append(line)
    index += 1
  
  # for label in labels:
  #   for i, y_true in enumerate(label):
  #     results[CLASS_NAMES[i]].append(y_true)

  with open(results_path, 'w') as f: 
    w = csv.DictWriter(f, line.keys())
    w.writeheader()
    w.writerows(results)

