import argparse
import config.config as configs
import torch

from run_experiment import run_active_learning_experiment
from models.bert import BertClassifier
from train import Learner
from evaluate import print_evaluation_report


def main():
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  config = configs.active_learning_config()
  results = run_active_learning_experiment(config, device)

  # model = BertClassifier(config=config.bert) 
  # model.to(device)
  # Train
  # learner = Learner(device, model, config.results_dir)
  # learner.train(config)

  # Test
  # save_model_path = os.path.join(config.results_dir, 'bert' + '.pth')
  # model.load_state_dict(torch.load(save_model_path), strict=False)
  # print_evaluation_report(model, config)


main()


