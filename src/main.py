from dotenv import load_dotenv
load_dotenv()

import argparse
import config.config as configs
import torch
import os
import active_learning.visualisation as al_vis
import pickle
import data_processing.utils as data_util
import run_experiment as experiment

from run_experiment import run_active_learning_experiment, Strategy
from models.bert import BertClassifier
from learner import Learner
from evaluate import print_evaluation_report
from active_learning.visualisation import plot_al_results
from data_processing.ev_parser import create_dataloader


def main():
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

  parser = argparse.ArgumentParser(description='Process some integers.')
  parser.add_argument('--experiment', type=str,
                      help='Experiment name')
  parser.add_argument('--al_strategy', type=str,
                      help='AL strategy to use e.g. RANDOM, MAX_ENTROPY, AVG_ENTROPY')
  parser.add_argument('--al_class', type=int, default=None,
                      help='Class which will guide AL.')
  parser.add_argument('--al_stratified', action='store_true',
                      help='Initial data batch will be created with stratified sampling.')

  args = parser.parse_args()
  
  if args.experiment == 'al':
    config = configs.active_learning_config()
    allowed_query_strategies = [s.name for s in Strategy]

    assert args.al_strategy in allowed_query_strategies, f'Please specify a valid AL strategy: {allowed_query_strategies}'
   
    config.query_strategy = args.al_strategy
    results = run_active_learning_experiment(
      config, 
      device=device,
      strategy_type=Strategy[args.al_strategy], 
      al_class=args.al_class, 
      first_sample_stratified=args.al_stratified)
  elif args.experiment == 'al_visualise':
    config = configs.active_learning_config()
    plot_al_results(
      strategies=['RANDOM', 'MAX_ENTROPY', 'CAL'], 
      config=config, 
      metrics=['accuracy', 'f1_score', 'train_time', 'sampling_emissions', 'sampling_time'])
    al_vis.plot_metrics_for_classes(
      config, 
      metrics=['f1_score', 'accuracy', 'precision', 'recall'], 
      classes=[0,1], 
      class_names=['functionality', 'range_anxiety'],
      strategies=['RANDOM', 'MAX_ENTROPY', 'CAL'])
  elif args.experiment == 'supervised':
    config = configs.multilabel_base()
    experiment.run_supervised_experiment(config, device, classes_to_track=[0,1])

    # Test
    # save_model_path = os.path.join(config.results_dir, 'bert' + '.pth')
    # model.load_state_dict(torch.load(save_model_path), strict=False)
    # print_evaluation_report(model, config)


main()
