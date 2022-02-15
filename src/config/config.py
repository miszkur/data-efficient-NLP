import ml_collections


def bert_config():
  config = ml_collections.ConfigDict()
  config.bert_version = 'small_bert/bert_en_uncased_L-2_H-128_A-2'
  config.lr = 5e-5 # 1e-4
  config.dropout = 0.2
  config.num_classes = 8
  return config

def multilabel_base():
  config = ml_collections.ConfigDict()
  config.epochs = 5  # 20

  config.batch_size = 8
  config.bert = bert_config()
  return config
