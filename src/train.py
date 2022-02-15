import torch.optim as optim
import torch.nn as nn
import torch
import os

from data_processing.ev_parser import create_dataloader
from config.config import multilabel_base
from tqdm.auto import tqdm


def train(config, model: nn.Module):
  config = multilabel_base()
  num_epochs = config.epochs
  
  save_mode_path = os.path.join('..', 'results', 'bert' + '.pth')

  trainloader = create_dataloader(config)
  validation_loader = create_dataloader(config, 'valid')
  model.train()
  optimizer = optim.AdamW(
    model.parameters(),
    lr=config.lr,
    weight_decay=config.weight_decay)
  criterion = nn.BCEWithLogitsLoss()

  for epoch_num in range(num_epochs):
    model.train()
    print('Epoch {}/{}'.format(epoch_num, num_epochs - 1))
    data_iterator = tqdm(trainloader, total=int(len(trainloader))) # ncols=70)
    running_loss = 0.0
    for batch_iter, batch in enumerate(data_iterator):
      # batch = {k: v.to(device) for k, v in batch.items()}
      # outputs = model(**batch)
      optimizer.zero_grad()
      # inputs = inputs.to(device, dtype=torch.float)
      # labels = labels.to(device, dtype=torch.float)
      output = model(input_id=batch['input_ids'], mask=batch['attention_mask'])

      loss = criterion(output, batch['label'])
      loss.backward()
      optimizer.step()

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
        output = model(input_id=batch['input_ids'], mask=batch['attention_mask'])
        val_loss += criterion(output, batch['label'])
    # total loss - divide by number of batches
    val_loss = val_loss / len(validation_loader)
    print('Validation Loss: {:.4f}'.format(val_loss))
  
  torch.save(model.state_dict(), save_mode_path)
