"""Script for backtranslating the dataset."""

from sklearn.utils import resample
import nlpaug.augmenter.word as naw
import config.config as configs
import os
import csv
import argparse
import pandas as pd

from data_processing.ev_parser import create_dataset, create_dataloader
from tqdm import tqdm

CLASS_NAMES = ['functionality','range_anxiety','availability','cost','ui','location','service_time','dealership']


def run_backtranslation(config):
  results_path = config.results_path
  aug = naw.BackTranslationAug()

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

    with open(results_path, 'w') as f: 
      w = csv.DictWriter(f, line.keys())
      w.writeheader()
      w.writerows(results)

def main():
  parser = argparse.ArgumentParser(description='Process some integers.')
  parser.add_argument('--visualise', action='store_true',
                      help='Visualise results of backtranslation.')
  parser.add_argument('--run', action='store_true',
                      help='Run backtranslation.')

  config = configs.backtranslation_config()
  args = parser.parse_args()

  if args.run:
    run_backtranslation(config)

  if args.visualise:
    results_path = config.results_path
    df = pd.read_csv(results_path)

    translation_redundant_frac = len(df[df.original == df.augmented]) / len(df)
    print(f'{100*translation_redundant_frac:.2f}%')
    print(df.original.duplicated().sum())

main()