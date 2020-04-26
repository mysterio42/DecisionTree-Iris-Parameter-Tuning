import argparse

import numpy as np

from utils.data import load_data
from utils.model import train_model
from utils.plot import graphviz


def parse_args():
    def str2bool(v):
        if isinstance(v, bool):
            return v
        if v.lower() in ('yes', 'true', 't', 'y', '1'):
            return True
        elif v.lower() in ('no', 'false', 'f', 'n', '0'):
            return False
        else:
            raise argparse.ArgumentTypeError('Boolean value expected')

    parser = argparse.ArgumentParser()
    parser.add_argument('--load', type=str2bool, default=False,
                        help='True: Load trained model  '
                             'False: Train model default: False ')
    parser.add_argument('--criterion', choices=['gini', 'entropy'], type=str,
                        help='Construct the Decision Tree based on entropy or gini')
    parser.add_argument('--method', choices=['cv', 'split'], type=str,
                        help='Cross-Validation or Train-Test-Split Method')
    parser.add_argument('--gs', type=str2bool, default=False,
                        help='Find optimal parameters with 10-Fold GridSearchCV')

    parser.print_help()

    return parser.parse_args()


if __name__ == '__main__':
    np.random.seed(1)

    args = parse_args()
    input()
    features, labels = load_data()

    model = train_model(features, labels, args)
