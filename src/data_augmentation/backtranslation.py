"""Script for backtranslating the dataset."""

import nlpaug.augmenter.word as naw
import sys
sys.path.append( '..' )
import config.config as configs
import os
import csv
import argparse
import pandas as pd
import torch
import string

from gramformer import Gramformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_selection import chi2
from sklearn.utils import resample
from data_processing.ev_parser import create_dataset, create_dataloader
from tqdm import tqdm

CLASS_NAMES = ['functionality','range_anxiety','availability','cost','ui','location','service_time','dealership']


def run_backtranslation(config):
  results_path = config.results_dir
  aug = naw.BackTranslationAug()

  train_df = pd.read_csv(os.path.join('..', '..', 'data', 'train_final.csv'))
  save_every_n = 10
  results = []
  for review in tqdm(train_df.itertuples(), total=train_df.shape[0]):
    review_text = review.review
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


def data_quality(df, cut_prop=1, filter_set = set()):
  train_Y = df[CLASS_NAMES]
  train_Y = train_Y.values
  vectorizer = CountVectorizer(lowercase=False)
  texts = df.review.tolist()
  train_X = vectorizer.fit_transform(texts)

  F, pvalues_f = chi2(train_X, train_Y)
  last = int(len(F) * cut_prop)
  sorted_F = sorted(zip(vectorizer.get_feature_names(), F), key=lambda x: x[1], reverse=True)[:last]
  values = [value for name, value in sorted_F if name not in filter_set]

  print(f'Data quality: {(sum(values) / len(values)):.2f}')

def gramatical_correctness(df):
  def set_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
      torch.cuda.manual_seed_all(seed)

  set_seed(1212)

  gf = Gramformer(models = 1, use_gpu=True) # 1=corrector, 2=detector
  
  sentences = df.review.tolist()
  influent_sentences_num = 0
  checked_sentences_num = 0
  for sentence in tqdm(sentences):
    if len(sentence.split()) > 55:
      continue

    checked_sentences_num += 1
    sentence_base = " ".join(sentence.split()) 
    corrected_sentences = gf.correct(sentence_base, max_candidates=1)
    sentence_base = sentence_base.translate(str.maketrans('', '', string.punctuation))
    for corrected_sentence in corrected_sentences:
      corrected_sentence = corrected_sentence.translate(str.maketrans('', '', string.punctuation))
      if sentence_base.lower() != corrected_sentence.lower():
        influent_sentences_num += 1
  influent_sentences_percent = influent_sentences_num*100/checked_sentences_num
  print(f'Number of incorrect sentences: {influent_sentences_num} ({influent_sentences_percent:.2f}%)')

def main():
  parser = argparse.ArgumentParser(description='Process some integers.')
  parser.add_argument('--visualise', action='store_true',
                      help='Visualise results of backtranslation.')
  parser.add_argument('--run', action='store_true',
                      help='Run backtranslation.')
  parser.add_argument('--convert', action='store_true',
                    help='Convert backtranslated csv to train set.')
  args = parser.parse_args()

  config = configs.backtranslation_config()
  results_path = config.results_path
  data_dir = os.path.join('..', '..', 'data')

  if args.run:
    run_backtranslation(config)

  if args.visualise:
    df = pd.read_csv(results_path)

    redundant_translations = len(df[df.original == df.augmented])
    redundant_translations_percent = 100 * redundant_translations / len(df)
    print(f'Redundant translations: {redundant_translations} {redundant_translations_percent:.2f}%')
    nan_translations = df.augmented.isna().sum()
    nan_translations_percent = 100 * nan_translations / len(df)
    print(f'NaN translations: {nan_translations} ({nan_translations_percent:.2f}%)')
    print(f"""Word "train": 
          original contains {df[df.original.str.contains('train')].shape[0]}, 
          augmented {df.augmented.str.contains('train').sum()}""")
    
    df_aug = pd.read_csv(os.path.join(data_dir, 'augmented_final.csv'))
    df_train = pd.read_csv(os.path.join(data_dir, 'train_final.csv'))

    # print('Train set')
    # gramatical_correctness(df_train)
    # print('Augmented set')
    # gramatical_correctness(df_aug)
    data_quality(df_train)
    data_quality(df_aug)


  if args.convert:
    df = pd.read_csv(results_path)
    df = df[~df.augmented.isna()]
    df = df[df.original != df.augmented]
    df.rename(columns={'augmented': 'review'}, inplace=True)
    df.drop(columns=['original'], inplace=True)
    df.to_csv(os.path.join(data_dir, 'augmented_final.csv'), index=False)

main()
