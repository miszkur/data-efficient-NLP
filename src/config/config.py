import ml_collections


def bert_tiny():
  config = ml_collections.ConfigDict()
  config.bert_version = 'prajjwal1/bert-tiny' #'bert-base-cased' 
  config.output_dim = 128
  config.dropout = 0.2
  config.num_classes = 8
  return config

def bert_config():
  config = ml_collections.ConfigDict()
  config.bert_version = 'bert-base-cased' 
  config.output_dim = 768
  config.dropout = 0.2
  config.num_classes = 8
  return config

def multilabel_base():
  config = ml_collections.ConfigDict()
  config.epochs = 2  # 20
  config.lr = 1e-4
  config.weight_decay = 0.01
  config.batch_size = 8
  config.warmup_steps = 500
  config.bert = bert_tiny()
  return config
