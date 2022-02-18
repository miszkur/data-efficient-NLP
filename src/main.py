import argparse
from data_processing.ev_parser import create_dataset, create_dataloader
from run_experiment import run_active_learning_experiment
import config.config as configs
from models.bert import BertClassifier
from train import Learner
from evaluate import print_evaluation_report
import torch
import os
import data_processing.ev_parser as ev
import numpy as np
from abc import abstractmethod
import seaborn as sns
import matplotlib.pyplot as plt
import pickle

def main():
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  results_dir = os.path.join('..', 'results')
  # config = multilabel_base()

  config = configs.active_learning_config()
  results = run_active_learning_experiment(config, device)

  sns.set_style('darkgrid')
  sns.lineplot(data=results, x="split", y="accuracy")
  plt.savefig(os.path.join(results_dir, 'accuracy.png'))

  al_results_path = os.path.join(results_dir, 'al', 'random.pkl')
  with open(al_results_path, 'wb') as fp:
    pickle.dump(results, fp)

  # model = BertClassifier(config=config.bert) 
  # model.to(device)
  # Train
  # learner = Learner(device, model, results_dir)
  # learner.train(config)

  # Test
  # save_model_path = os.path.join(results_dir, 'bert' + '.pth')
  # model.load_state_dict(torch.load(save_model_path), strict=False)
  # print_evaluation_report(model, config)


main()


