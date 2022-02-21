from data_processing.ev_parser import create_dataloader
from torchmetrics import HammingDistance, F1Score
from sklearn.metrics import classification_report
from tqdm import tqdm

import torch.nn as nn
import numpy as np
import torch



hamming_distance = HammingDistance()
f1 = F1Score(num_classes=8)


def compute_accuracy_and_f1(output, labels):
  """Compute accuracy accprding to the following formula:
  accuracy = 1 - hamming distance"""
  with torch.no_grad():
    preds = torch.sigmoid(output).cpu()
    target = labels.cpu().to(torch.int)
    acc = hamming_distance(preds, target)
    return 1 - acc.item(), f1(preds, target)

def print_evaluation_report(model, config, test_save_path=None):
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  test_loader, target_names = create_dataloader(config, 'test', return_target_names=True)

  model.eval()
  criterion = nn.BCEWithLogitsLoss()
  loss = 0
  accuracy = 0
  f1_score = 0
  y_true = []
  y_pred = []
  with torch.no_grad():
    for batch in test_loader:
      inputs = batch['input_ids'].to(device, dtype=torch.long)
      attention_masks = batch['attention_mask'].to(device, dtype=torch.long)
      token_type_ids = batch['token_type_ids'].to(device, dtype=torch.long)
      labels = batch['label'].to(device, dtype=torch.float)

      output = model(input_ids=inputs, attn_mask=attention_masks, token_type_ids=token_type_ids)
      loss = criterion(output, labels)
      loss += loss.item()
      acc, f1_s = compute_accuracy_and_f1(output, labels)
      accuracy += acc
      f1_score += f1_s
      
      y_true.append(labels.cpu().numpy())
      y_pred.append(np.round(torch.sigmoid(output).cpu().numpy()))

  loss /= len(test_loader)
  accuracy /= len(test_loader)
  f1_score /= len(test_loader)
  print('\nPerformance on the test set')
  print(f'Test loss: {loss:.4f}, accuracy: {accuracy:.4f}, F1-score: {f1_score:.4f}')

  print('Per-class metrics')
  y_true = np.concatenate(y_true)
  y_pred = np.concatenate(y_pred)  
  print(classification_report(y_true, y_pred, target_names=target_names, zero_division=0))

  print('Per-class accuracy')
  correctly_classified = (y_pred == y_true)
  acc = correctly_classified.sum(axis=0) / y_pred.shape[0]
  for i, name in enumerate(target_names):
    print(f'{name}: {(acc[i]):.2f}')

   
