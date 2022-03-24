import ml_collections
import os


def bert_tiny():
  config = ml_collections.ConfigDict()
  config.bert_version = 'prajjwal1/bert-tiny' #'bert-base-cased' 
  config.output_dim = 128
  config.dropout = 0.1
  config.num_classes = 8
  return config

def bert_config():
  config = ml_collections.ConfigDict()
  config.bert_version = 'bert-base-cased' 
  config.output_dim = 768
  config.dropout = 0.1
  config.num_classes = 8
  return config

def multilabel_base():
  config = ml_collections.ConfigDict()
  config.num_epochs = 10  # 20
  config.lr = 5e-5
  config.weight_decay = 0.01
  config.batch_size = 8
  config.warmup_steps = 500
  config.max_grad_norm = 1.0
  config.bert = bert_config()
  config.results_dir = os.path.join('..', 'results')
  return config

def active_learning_config():
  config = multilabel_base()
  config.seeds = [2, 42, 52, 62, 72]
  config.num_al_iters = 17
  config.sample_size = 48
  config.results_dir = os.path.join('..', 'results', 'al')
  config.query_strategy = 'random'
  return config


def zero_shot_config():
  config = multilabel_base()
  config.bart = 'facebook/bart-large-mnli'
  config.class_names = ['functionality', 'range anxiety', 'availability', 'cost', 'ui', 'location', 'service time', 'dealership']
  # modify class names: 
  # config.class_names = ['functionality', 'trip', 'availability', 'payment', 'user interactions', 'location', 'charging time', 'dealership']
  # config.class_names = [
  #   'functionality', 'range', 'number of stations available', 'parking charging', 'user interactions', 'general location', 'charging rate', '	dealership charging experience']
  config.results_dir = os.path.join('..', 'results', 'zero-shot')
  return config
