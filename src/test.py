from data_processing.ev_parser import create_dataloader
from tqdm import tqdm
import torch.nn as nn

def inference(model, test_save_path=None):
    testloader = create_dataloader(1, 'test')

    model.eval()
    bce_loss = nn.BCEWithLogitsLoss()
    metric_list = 0.0
    for batch in tqdm(testloader):
      output = model(input_id=batch['input_ids'], mask=batch['attention_mask'])
      loss = bce_loss(output, batch['label'])

    model.eval() 
    val_loss = 0
    with torch.no_grad():
      for batch in validation_loader:
        output = model(input_id=batch['input_ids'], mask=batch['attention_mask'])
        loss += criterion(output, batch['label'])
    # total loss - divide by number of batches
    val_loss = loss / len(validation_loader)
    print('Validation Loss: {:.4f}'.format(val_loss))
