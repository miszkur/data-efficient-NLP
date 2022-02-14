import tensorflow as tf
import pandas as pd
import os

DATA_DIR = 'data'
AUTOTUNE = tf.data.experimental.AUTOTUNE
SHUFFLE_BUFFER_SIZE = 1024

def load_dataset(split):
  df = pd.read_csv(os.path.join(DATA_DIR, f'{split}_final.csv'))

  reviews = df.review.to_list()

  labels = []
  for x in df.itertuples():
    labels.append(list(x[3:]))

  label_names = df.columns[2:]
  return reviews, labels, label_names


def create_dataset(batch_size, is_training=True, split='train'):
  """Load and parse dataset.
  Args:
      filenames: list of image paths
      labels: numpy array of shape (BATCH_SIZE, N_LABELS)
      is_training: boolean to indicate training mode
  """
  
  assert split in ['train', 'test', 'valid']

  reviews, labels, label_names = load_dataset(split)

  dataset = tf.data.Dataset.from_tensor_slices((reviews, labels))
  if is_training == True:
      # This is a small dataset, only load it once, and keep it in memory.
      dataset = dataset.cache()
      # Shuffle the data each buffer size
      dataset = dataset.shuffle(buffer_size=SHUFFLE_BUFFER_SIZE)
      
  # Batch the data for multiple steps
  dataset = dataset.batch(batch_size)
  # Fetch batches in the background while the model is training.
  dataset = dataset.prefetch(buffer_size=AUTOTUNE)
  
  return dataset, label_names
