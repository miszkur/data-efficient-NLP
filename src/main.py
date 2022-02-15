import json
import argparse
from data_processing.ev_parser import create_dataset
from config.config import multilabel_base


def main():
    config = multilabel_base()
    ds_train = create_dataset(batch_size=config.batch_size)
    print(ds_train)

main()


