import argparse
import config.config as configs
import torch
import os

from run_experiment import run_active_learning_experiment, Strategy
from models.bert import BertClassifier
from learner import Learner
from evaluate import print_evaluation_report
from active_learning.visualisation import plot_al_results


def main():
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

  parser = argparse.ArgumentParser(description='Process some integers.')
  parser.add_argument('--experiment', type=str,
                      help='Experiment name')
  parser.add_argument('--al_strategy', type=str,
                      help='AL strategy to use e.g. RANDOM, MAX_ENTROPY, AVG_ENTROPY')
  args = parser.parse_args()

  if args.experiment == 'al':
    config = configs.active_learning_config()
    allowed_query_strategies = [s.name for s in Strategy]
    
    assert args.al_strategy in allowed_query_strategies, f'Please specify a valid AL strategy: {allowed_query_strategies}'
    
    config.query_strategy = args.al_strategy
    results = run_active_learning_experiment(config, device, Strategy[args.al_strategy])
    plot_al_results(['RANDOM', 'MAX_ENTROPY', 'AVG_ENTROPY'], config)
  elif args.experiment == 'supervised':
    config = configs.multilabel_base()
    model = BertClassifier(config=config.bert) 
    model.to(device)
    # Train
    learner = Learner(device, model, config.results_dir)
    learner.train(config)

    # Test
    save_model_path = os.path.join(config.results_dir, 'bert' + '.pth')
    model.load_state_dict(torch.load(save_model_path), strict=False)
    print_evaluation_report(model, config)


main()
