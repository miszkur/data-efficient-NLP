import torch.optim as optim
import torch.nn as nn
import numpy as np
import torch
import os

from data_processing.ev_parser import create_dataloader
from config.config import multilabel_base
from tqdm.auto import tqdm
from torchmetrics import HammingDistance
from visualisation.utils import plot_history 

from transformers import get_cosine_schedule_with_warmup

hamming_distance = HammingDistance()

def compute_accuracy(output, labels):
  """Compute accuracy accprding to the following formula:
  accuracy = 1 - hamming distance"""
  with torch.no_grad():
    preds = torch.sigmoid(output).cpu()
    target = labels.cpu().to(torch.int)
    acc = hamming_distance(preds, target)
    return 1 - acc.item()

def train(config, model: nn.Module, results_dir):
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

  train_loader = create_dataloader(config)
  validation_loader = create_dataloader(config, 'valid')

  num_epochs = config.epochs
  num_training_steps = num_epochs * len(train_loader)

  history = {'loss': [], 'val_loss': [], 'accuracy': [], 'val_accuracy': []}
  
  model.train()
  optimizer = optim.AdamW(
    model.parameters(),
    lr=config.lr,
    weight_decay=config.weight_decay)
  criterion = nn.BCEWithLogitsLoss()
  # lr_scheduler = get_cosine_schedule_with_warmup(
  #   optimizer,
  #   num_warmup_steps=config.warmup_steps, 
  #   num_training_steps=num_training_steps)

  for epoch_num in range(num_epochs):
    model.train()
    print('Epoch {}/{}'.format(epoch_num, num_epochs - 1))
    data_iterator = tqdm(train_loader, total=int(len(train_loader))) # ncols=70)
    running_loss = 0.0
    accuracy = 0.0
    for batch_iter, batch in enumerate(data_iterator):
      optimizer.zero_grad()
      inputs = batch['input_ids'].to(device, dtype=torch.long)
      attention_masks = batch['attention_mask'].to(device, dtype=torch.long)
      token_type_ids = batch['token_type_ids'].to(device, dtype=torch.long)
      labels = batch['label'].to(device, dtype=torch.float)
      output = model(input_ids=inputs, attn_mask=attention_masks, token_type_ids=token_type_ids)

      loss = criterion(output, labels)
      loss.backward()
      optimizer.step()
      # lr_scheduler.step()

      # Calculate statistics
      current_accuracy = compute_accuracy(output, labels)
      accuracy += current_accuracy
      running_loss += loss.item()
      data_iterator.set_postfix(loss=(loss.item()), accuracy=(current_accuracy))
    
    train_loss = running_loss / len(train_loader)
    train_accuracy = accuracy / len(train_loader)
    
    model.eval() 
    val_loss = 0
    val_accuracy = 0
    with torch.no_grad():
      for batch in validation_loader:
        inputs = batch['input_ids'].to(device, dtype=torch.long)
        attention_masks = batch['attention_mask'].to(device, dtype=torch.long)
        token_type_ids = batch['token_type_ids'].to(device, dtype=torch.long)
        labels = batch['label'].to(device, dtype=torch.float)
        output = model(input_ids=inputs, attn_mask=attention_masks, token_type_ids=token_type_ids)
        loss = criterion(output, labels)
        val_loss += loss.item()
        val_accuracy += compute_accuracy(output, labels)

    val_loss /= len(validation_loader)
    val_accuracy /= len(validation_loader)
    print(f'Training loss: {train_loss:.4f}, Training accuracy: {train_accuracy:.4f}')
    print(f'Validation Loss: {val_loss:.4f},  Validation accuracy: {val_accuracy:.4f}')

    history['loss'].append(train_loss)
    history['val_loss'].append(val_loss)
    history['val_accuracy'].append(val_accuracy)
    history['accuracy'].append(train_accuracy)
  
  save_model_path = os.path.join(results_dir, 'bert' + '.pth')
  torch.save(model.state_dict(), save_model_path)
  plot_history(history_dict=history, results_dir=results_dir)
