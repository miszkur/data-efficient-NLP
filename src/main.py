import json
import argparse
from data_processing.ev_parser import create_dataset
from config.config import multilabel_base
from models.bert import BertClassifier
from train import train

def main():
    config = multilabel_base()
    model = BertClassifier(config=config.bert) 
    train(config=config, model=model)

main()


