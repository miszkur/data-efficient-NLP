
from torch import nn
from transformers import BertModel

class BertClassifier(nn.Module):

  def __init__(self, config):
    super(BertClassifier, self).__init__()
    self.config = config
    self.bert = BertModel.from_pretrained(config.bert_version)
    self.dropout = nn.Dropout(config.dropout)
    self.linear = nn.Linear(config.output_dim, config.num_classes)
    self.relu = nn.ReLU()

  def forward(self, input_id, mask):
    # pooled_output - CLS token
    _, pooled_output = self.bert(input_ids=input_id, attention_mask=mask, return_dict=False)
    dropout_output = self.dropout(pooled_output)
    linear_output = self.linear(dropout_output)

    return linear_output
