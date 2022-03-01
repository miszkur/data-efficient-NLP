from transformers import BertTokenizer
from torch.utils.data import Dataset, DataLoader, random_split
from ml_collections import ConfigDict

import torch
import numpy as np
import pandas as pd
import os

DATA_DIR = '../data'


class EvDataset(Dataset):
  def __init__(self, df, tokenizer):
    self.tokenizer = tokenizer

    self.targets = []
    for x in df.itertuples():
      self.targets.append(list(x[3:]))

    self.target_names = df.columns[2:]
    self.reviews = df.review

  def __len__(self):
    return len(self.targets)

  def __getitem__(self, item):
    review = str(self.reviews[item])
    target = self.targets[item]
    encoding = self.tokenizer.encode_plus(
      review,
      add_special_tokens=True, # Add [CLS] [SEP] tokens
      return_token_type_ids=True,
      padding='max_length',
      max_length=512, 
      truncation=True,
      return_attention_mask=True,
      return_tensors='pt',
    )

    input_ids = encoding['input_ids'].flatten()
    attn_mask = encoding['attention_mask'].flatten()
    token_type_ids = encoding["token_type_ids"].flatten()

    return {
      'review_text': review,
      'input_ids': input_ids,
      'attention_mask': attn_mask,
      'token_type_ids': token_type_ids,
      'label': torch.tensor(target, dtype=torch.float)
    }

def create_dataset(split='train') -> Dataset:
  """Load and parse dataset.

  Args:
      split (str, optional): Which data split to load, one of: ['train', 'test', 'valid'].
      Defaults to 'train'.

  Returns:
      Dataset: EV dataset.
  """
  assert split in ['train', 'test', 'valid']

  df = pd.read_csv(os.path.join(DATA_DIR, f'{split}_final.csv'))
  tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
  return EvDataset(df, tokenizer)

def create_dataloader(
  config: ConfigDict, 
  split='train', 
  return_target_names=False) -> DataLoader:
  """Load dataset and wrap it in DataLoader.

  Args:
      config (ConfigDict): configuration dictionary.
      split (str, optional): Which data split to load, one of: ['train', 'test', 'valid'].
      Defaults to 'train'.
      return_target_names (bool, optional): If true, also target names 
      (names of topics) will be returned. Defaults to False.

  Returns:
      DataLoader: EV dataset's dataloader.
      Optionally returns also target names.
  """
  
  assert split in ['train', 'test', 'valid']
  is_training = (split == 'train')

  df = pd.read_csv(os.path.join(DATA_DIR, f'{split}_final.csv'))
  
  tokenizer = BertTokenizer.from_pretrained(config.bert.bert_version)
  ds = EvDataset(df, tokenizer)

  if split == 'valid': 
    # Divide validation set into validation and test set.
    valid, test = random_split(
      ds, [len(ds)//2, len(ds) - len(ds)//2], generator=torch.Generator().manual_seed(42))
    dl = (
      DataLoader(
      valid, 
      batch_size=config.batch_size,
      shuffle=False),
      DataLoader(
      test, 
      batch_size=config.batch_size,
      shuffle=False)
    )
  else:
    dl = DataLoader(
      ds, 
      batch_size=config.batch_size,
      shuffle=is_training)

  if return_target_names:
    return dl, ds.target_names

  return dl

