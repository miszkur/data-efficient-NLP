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
  config.use_aug_data = False
  return config

def active_learning_config():
  config = multilabel_base()
  config.seeds = [2, 42, 52, 62, 72]
  config.num_al_iters = 17
  config.sample_size = 48
  config.results_dir = os.path.join('..', 'results', 'al')
  config.query_strategy = 'random'
  return config

def al_aug_config():
  config = active_learning_config()
  config.data_dir = os.path.join('..', 'data')
  config.results_dir = os.path.join('..', 'results', 'al_aug')
  config.sample_size = 48
  config.num_al_iters = 17
  config.query_strategy = 'CAL'
  config.use_aug_data = True
  return config

def zero_shot_config():
  config = multilabel_base()
  config.bart = 'facebook/bart-large-mnli'
  config.class_names = ['functionality', 'range anxiety', 'availability', 'cost', 'ui', 'location', 'service time', 'dealership']
  config.results_dir = os.path.join('..', 'results', 'zero-shot')
  return config

def backtranslation_config():
  config = multilabel_base()
  config.results_path = os.path.join('..', '..', 'results', 'backtranslation', 'augmented.csv')
  return config


def augmentation_config():
  config = multilabel_base()
  config.results_dir = os.path.join('..', 'results', 'augmentation')
  config.data_dir = os.path.join('..', 'data')
  config.seeds = [2, 42, 52, 62, 72]
  return config

def augmentation_al_data_config():
  config = augmentation_config()
  config.sample_size = 48
  config.num_al_iters = 17
  config.query_strategy = 'CAL'
  return config
  
def few_shot_config():
  config = multilabel_base()
  config.n_way = 3
  config.k_shot = 5
  config.lr = 1e-4
  return config
