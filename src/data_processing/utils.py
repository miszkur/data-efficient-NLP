import numpy as np

def print_label_stats(data_loader):
  labels = np.array([0.0 for i in range(8)])
  for batch in data_loader:
    labels += batch['label'].sum(axis=0).cpu().numpy()
  print(f'Label counts: {labels}')