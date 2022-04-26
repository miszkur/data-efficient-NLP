
from torch import nn
from transformers import BertModel, DistilBertModel
import torch

class PrototypicalNetwork(nn.Module):
  def __init__(self, config, device):
    super(PrototypicalNetwork, self).__init__()
    self.config = config
    self.metric = 'euclidean' # TODO: move to config
    bert_config = config.bert
    self.bert = BertModel.from_pretrained(bert_config.bert_version, return_dict=True)
    self.dropout = nn.Dropout(bert_config.dropout)
    self.distance_layer = nn.Linear(bert_config.output_dim, bert_config.num_classes)
    self.relu = nn.LeakyReLU(0.3)
    self.device = device
    self.n_way = config.n_way
    self.k_shot = config.k_shot

  def embed_batch(self, episode):
    input_ids = []
    attention_mask = []
    token_type_ids = []
    for class_samples in episode: # N-way
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
      token_type_ids=token_type_ids
      )
    
    # relu_output = self.relu(embedding.pooler_output)
    # dropout_output = self.dropout(relu_output)
    # results.append(embedding.pooler_output)
    return embedding.pooler_output

  def euclidean_dist(self, x, y):
    # x: n_way x output_dim
    # y: n_query x output_dim
    n = x.size(0)
    m = y.size(0)
    d = x.size(1)
    assert d == y.size(1)

    x = x.unsqueeze(1).expand(n, m, d)
    y = y.unsqueeze(0).expand(n, m, d)

    return torch.pow(x - y, 2).sum(2)

  def forward(self, episode):
    support_positive = episode['xs'] # n_way x k_shots
    support_negative = episode['xn']
    queries = episode['xq']

    positive_out = self.embed_batch(support_positive)
    negative_out = self.embed_batch(support_negative)
    query_out = self.embed_batch(queries)

    z_dim = positive_out.size(-1)
    means_positive = positive_out.view(self.n_way, self.k_shot, z_dim).mean(dim=[1])
    means_negative = negative_out.view(self.n_way, self.k_shot, z_dim).mean(dim=[1])
    
    means_positive = nn.functional.normalize(means_positive, dim=1)
    means_negative = nn.functional.normalize(means_negative, dim=1)
    query_out = nn.functional.normalize(query_out, dim=1)

    if self.metric == 'euclidean':
      dist_pos = self.euclidean_dist(means_positive, query_out) # n_way x n_query
      dist_neg = self.euclidean_dist(means_negative, query_out) 
    
    dist_pos_rescale = torch.exp(-dist_pos) 
    dist_neg_rescale = torch.exp(-dist_neg) 

    probabilities = torch.div(
      dist_pos_rescale, torch.add(dist_pos_rescale, dist_neg_rescale))

    return torch.transpose(probabilities, 0, 1)
