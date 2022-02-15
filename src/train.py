import torch.optim as optim
import torch.nn as nn
import torch
import os

from data_processing.ev_parser import create_dataloader
from config.config import multilabel_base
from tqdm.auto import tqdm

from transformers import get_cosine_schedule_with_warmup


def train(config, model: nn.Module):
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

  config = multilabel_base()
  num_epochs = config.epochs
  
  save_mode_path = os.path.join('..', 'results', 'bert' + '.pth')

  trainloader = create_dataloader(config)
  validation_loader = create_dataloader(config, 'valid')

  num_training_steps = num_epochs * len(trainloader)
  
  model.train()
  optimizer = optim.AdamW(
    model.parameters(),
    lr=config.lr,
    weight_decay=config.weight_decay)
  criterion = nn.BCEWithLogitsLoss()
  lr_scheduler = get_cosine_schedule_with_warmup(
    optimizer,
    num_warmup_steps=config.warmup_steps, 
    num_training_steps=num_training_steps)

  for epoch_num in range(num_epochs):
    model.train()
    print('Epoch {}/{}'.format(epoch_num, num_epochs - 1))
    data_iterator = tqdm(trainloader, total=int(len(trainloader))) # ncols=70)
    running_loss = 0.0
    for batch_iter, batch in enumerate(data_iterator):
      optimizer.zero_grad()
      inputs = batch['input_ids'].to(device, dtype=torch.long)
      attention_masks = batch['attention_mask'].to(device, dtype=torch.float)
      labels = batch['label'].to(device, dtype=torch.float)
      output = model(input_id=inputs, mask=attention_masks)

      loss = criterion(output, labels)
      loss.backward()
      optimizer.step()
      lr_scheduler.step()

      # print statistics
      running_loss += loss.item()
      data_iterator.set_postfix(
        loss=(loss.item()))
    
    epoch_loss = running_loss / len(trainloader)
    print('Training Loss: {:.4f}'.format(epoch_loss))
    
    model.eval() 
    val_loss = 0
    with torch.no_grad():
      for batch in validation_loader:
        inputs = batch['input_ids'].to(device, dtype=torch.long)
        attention_masks = batch['attention_mask'].to(device, dtype=torch.float)
        labels = batch['label'].to(device, dtype=torch.float)
        output = model(input_id=inputs, mask=attention_masks)
        val_loss += criterion(output, labels)
    val_loss = val_loss / len(validation_loader)
    print('Validation Loss: {:.4f}'.format(val_loss))
  
  torch.save(model.state_dict(), save_mode_path)