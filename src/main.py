import json
import argparse
from data_processing.ev_parser import create_dataset
from config.config import multilabel_base
from models.bert import BertClassifier
from train import train
from evaluate import evaluate
import torch
import os

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    results_dir = os.path.join('..', 'results')
    config = multilabel_base()
    model = BertClassifier(config=config.bert) 
    model.to(device)
    train(config=config, model=model, results_dir=results_dir)
    save_model_path = os.path.join(results_dir, 'bert' + '.pth')

    model.load_state_dict(torch.load(save_model_path), strict=False)
    evaluate(model, config)

main()


