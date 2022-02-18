
from torch import nn
from transformers import BertModel

class BertClassifier(nn.Module):

  def __init__(self, config):
    super(BertClassifier, self).__init__()
    self.config = config
    self.bert = BertModel.from_pretrained(config.bert_version, return_dict=True)
    self.dropout = nn.Dropout(config.dropout)
    self.linear = nn.Linear(config.output_dim, config.num_classes)
    self.relu = nn.ReLU()

  def forward(self, input_ids, attn_mask, token_type_ids):
    output = self.bert(
      input_ids, 
      attention_mask=attn_mask, 
      token_type_ids=token_type_ids)
    
    # pooler_output - CLS token
    dropout_output = self.dropout(output.pooler_output)
    linear_output = self.linear(dropout_output)

    return linear_output
