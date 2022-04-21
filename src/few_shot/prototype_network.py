
from torch import nn
from transformers import BertModel, DistilBertModel
import torch

class ProtoypeNetwork(nn.Module):
  def __init__(self, config, device):
    super(ProtoypeNetwork, self).__init__()
    self.config = config
    # self.bert = DistilBertModel.from_pretrained('distilbert-base-uncased', return_dict=True)
    self.bert = BertModel.from_pretrained(config.bert_version, return_dict=True)
    self.dropout = nn.Dropout(config.dropout)
    self.distance_layer = nn.Linear(config.output_dim, config.num_classes)
    self.relu = nn.LeakyReLU(0.3)
    self.device = device

  def embed_batch(self, episode):
    results = []
    for class_samples in episode: # N-way
      input_ids = []
      attention_mask = []
      token_type_ids = []
      for sample in class_samples: # K-shot
        input_ids.append(sample['input_ids'])
        attention_mask.append(sample['attention_mask'])
        token_type_ids.append(sample['token_type_ids'])

      input_ids = torch.stack(input_ids).to(self.device)
      attention_mask = torch.stack(attention_mask).to(self.device)
      token_type_ids = torch.stack(token_type_ids).to(self.device)
      
      embedding = self.bert(
        input_ids, 
        attention_mask=attention_mask, 
        token_type_ids=token_type_ids)
      
      relu_output = self.relu(embedding.pooler_output)
      dropout_output = self.dropout(relu_output)
      dist_output = self.distance_layer(dropout_output)
      results.append(dist_output)
    return results

  def forward(self, episode):
    support_positive = episode['xs'] # n_classes x k_shots
    support_negative = episode['xn']
    queries = episode['xq']

    positive_out = self.embed_batch(support_positive)
    negative_out = self.embed_batch(support_negative)
    query_out = self.embed_batch(queries)
    
    return 0
