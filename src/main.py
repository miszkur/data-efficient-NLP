from dotenv import load_dotenv
load_dotenv()

import argparse
import config.config as configs
import torch
import os
import active_learning.visualisation as al_vis
import pickle
import data_processing.utils as data_util

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
    # Visualisation
    # al_vis.cold_vs_warm_start('AVG_ENTROPY', config)
    # plot_al_results(['RANDOM', 'AVG_ENTROPY', 'MAX_ENTROPY', 'CAL'], config)
    # al_vis.plot_metrics_for_classes(config, ['f1_score', 'accuracy', 'precision', 'recall'], [0,1], ['RANDOM', 'MAX_ENTROPY'])
  elif args.experiment == 'supervised':
    config = configs.multilabel_base()
    valid_loader, test_loader = create_dataloader(config, 'valid')
    results = {'accuracy': [], 'f1_score':[]}
    for _ in range(5):
      model = BertClassifier(config=config.bert) 
      model.to(device)
      # Train
      learner = Learner(device, model, config.results_dir)
      learner.train(config, validation_loader=valid_loader)
      
      # Evaluate
      loss, accuracy, f1_score = learner.evaluate(test_loader)
      print(f'Test loss: {loss}, accuracy: {accuracy}, f1 score: {f1_score}')
      results['accuracy'].append(accuracy)
      results['f1_score'].append(f1_score)

      print('Saving results..')
      results_path = os.path.join(config.results_dir, f'SUPERVISED.pkl')
      with open(results_path, 'wb') as fp:
        pickle.dump(results, fp)
    # Test
    # save_model_path = os.path.join(config.results_dir, 'bert' + '.pth')
    # model.load_state_dict(torch.load(save_model_path), strict=False)
    # print_evaluation_report(model, config)


main()
