from transformers import BertTokenizer
from torch.utils.data import Dataset, DataLoader
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
    encoding = self.tokenizer.encode(
      review,
      add_special_tokens=True, # Add [CLS] [SEP] tokens
      return_token_type_ids=False,
      padding='max_length',
      return_attention_mask=True,
      return_tensors='pt',
    )

    return {
      'review_text': review,
      'input_ids': encoding['input_ids'].flatten(),
      'attention_mask': encoding['attention_mask'].flatten(),
      'targets': torch.tensor(target, dtype=torch.long)
    }

def create_dataset(split='train'):
  """Load and parse dataset."""
  assert split in ['train', 'test', 'valid']

  df = pd.read_csv(os.path.join(DATA_DIR, f'{split}_final.csv'))
  tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
  return EvDataset(df, tokenizer)

def create_dataloader(batch_size, split='train'):
  """Load and parse dataset.
  Args:
      filenames: list of image paths
      labels: numpy array of shape (BATCH_SIZE, N_LABELS)
      is_training: boolean to indicate training mode
  """
  
  assert split in ['train', 'test', 'valid']
  is_training = (split == 'train')

  df = pd.read_csv(os.path.join(DATA_DIR, f'{split}_final.csv'))
  
  tokenizer = BertTokenizer.from_pretrained('bert-base-cased')

  return DataLoader(
    EvDataset(df, tokenizer), 
    batch_size=batch_size,
    shuffle=is_training)

