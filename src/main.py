from dotenv import load_dotenv
load_dotenv()

import argparse
import config.config as configs
import torch
import active_learning.visualisation as al_vis
import visualisation.zero_shot as zs_vis
import run_al_experiment as experiment

from run_al_experiment import run_active_learning_experiment, Strategy
from run_zero_shot_experiment import run_zero_shot_experiment
from active_learning.visualisation import plot_al_results


def main():
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

  parser = argparse.ArgumentParser(description='Process some integers.')
  parser.add_argument('--experiment', type=str,
                      help='Experiment name')
  parser.add_argument('--al_strategy', type=str,
                      help='AL strategy to use e.g. RANDOM, MAX_ENTROPY, AVG_ENTROPY')
  parser.add_argument('--al_stratified', action='store_true',
                      help='Initial data batch will be created with stratified sampling.')
  parser.add_argument('--visualise', action='store_true',
                      help='Visualise results of the experiment.')

  args = parser.parse_args()

  if args.experiment == 'zero-shot':

    config = configs.zero_shot_config()

    if args.visualise:
      zs_vis.visualise_per_class_performance(config)
    else:

  elif args.experiment == 'al':
    config = configs.active_learning_config()

    if args.visualise:
      strategies = ['RANDOM', 'MAX_ENTROPY', 'CAL']
      plot_al_results(
        strategies=strategies, 
        config=config, 
        metrics=['accuracy', 'f1_score', 'train_time', 'sampling_emissions', 'sampling_time'])
      al_vis.plot_metrics_for_classes(
        config, 
        metrics=['f1_score', 'accuracy', 'precision', 'recall'], 
        classes=[0,1], 
        class_names=['functionality', 'range_anxiety'],
        strategies=strategies)
    else: 
      allowed_query_strategies = [s.name for s in Strategy]

      assert args.al_strategy in allowed_query_strategies, f'Please specify a valid AL strategy: {allowed_query_strategies}'
    
      config.query_strategy = args.al_strategy
      results = run_active_learning_experiment(
        config, 
        device=device,
        strategy_type=Strategy[args.al_strategy], 
        first_sample_stratified=args.al_stratified)
  elif args.experiment == 'al_visualise':
    config = configs.active_learning_config()

  elif args.experiment == 'supervised':
    config = configs.multilabel_base()
    experiment.run_supervised_experiment(config, device, classes_to_track=[0,1])

    # Test
    # save_model_path = os.path.join(config.results_dir, 'bert' + '.pth')
    # model.load_state_dict(torch.load(save_model_path), strict=False)
    # print_evaluation_report(model, config)


main()
