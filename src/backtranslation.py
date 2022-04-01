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

  first_path = os.path.join('..', 'results', 'backtranslation', 'first.csv')
  df_first = pd.read_csv(first_path)

  train_df = pd.read_csv(os.path.join('..', 'data', 'train_final.csv'))
  save_every_n = 10
  results = []
  for review in tqdm(train_df.itertuples(), total=train_df.shape[0]):
    review_text = review.review
    
    if (df_first.original == review_text).any():
      # print(df_first[df_first.original == review_text].iloc[0].augmented)
      augmented_text = df_first[df_first.original == review_text].iloc[0].augmented
    else:
      augmented_text = aug.augment(review_text)

    index = review.Index
    line = {
      'id': index,
      'original': review_text,
      'augmented': augmented_text,
      'functionality': review.functionality,
      'range_anxiety': review.range_anxiety,
      'availability': review.availability,
      'cost': review.cost,
      'ui': review.ui,
      'location': review.location,
      'service_time': review.service_time,
      'dealership': review.dealership
    }
    results.append(line)

    if index % save_every_n == 0:
      with open(results_path, 'a') as f: 
        w = csv.DictWriter(f, line.keys())
        # w.writeheader()
        w.writerows(results)
        results = []

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