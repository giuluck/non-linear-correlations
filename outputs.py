import argparse
import logging
import os
import re
import warnings

from experiments import LearningExperiment
from items.datasets import Communities, Adult, Census
from items.hgr import DoubleKernelHGR, SingleKernelHGR, AdversarialHGR, DensityHGR

# noinspection DuplicatedCode
os.environ['WANDB_SILENT'] = 'true'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings("ignore", ".*does not have many workers.*")
for name in ["lightning_fabric", "pytorch_lightning.utilities.rank_zero", "pytorch_lightning.accelerators.cuda"]:
    log = logging.getLogger(name)
    log.propagate = False
    log.setLevel(logging.ERROR)


# function to retrieve the valid metric
def metrics(key):
    if key == 'nn':
        return 'HGR-NN', AdversarialHGR()
    elif key == 'kde':
        return 'HGR-KDE', DensityHGR()
    elif key == 'kb':
        return 'HGR-KB', DoubleKernelHGR()
    elif key == 'sk':
        return 'HGR-SK', SingleKernelHGR()
    elif re.compile('kb-([0-9]+)').match(key):
        degree = int(key[3:])
        return f'HGR-KB ({degree})', DoubleKernelHGR(degree_a=degree, degree_b=degree)
    elif re.compile('sk-([0-9]+)').match(key):
        degree = int(key[3:])
        return f'HGR-SK ({degree})', SingleKernelHGR(degree=degree)
    else:
        raise KeyError(f"Invalid key '{key}' for metric")


# list all the valid datasets
datasets = dict(
    communities=Communities(),
    adult=Adult(),
    census=Census()
)

# build argument parser
parser = argparse.ArgumentParser(description='Train multiple neural networks using different HGR metrics as penalizers')
parser.add_argument(
    '-f',
    '--folder',
    type=str,
    default='results',
    help='the path where to search and store the results and the exports'
)
parser.add_argument(
    '-d',
    '--datasets',
    type=str,
    nargs='+',
    choices=list(datasets),
    default=['census', 'communities', 'adult'],
    help='the datasets on which to run the experiment'
)
parser.add_argument(
    '-m',
    '--metrics',
    type=str,
    nargs='*',
    default=['kb', 'sk', 'nn'],
    help='the metrics used as penalties'
)
parser.add_argument(
    '-s',
    '--steps',
    type=int,
    default=500,
    help='the number of steps to run for each network'
)
parser.add_argument(
    '-k',
    '--folds',
    type=int,
    default=5,
    help='the number of folds to be used for cross-validation'
)
parser.add_argument(
    '-u',
    '--units',
    type=int,
    nargs='*',
    help='the hidden units of the neural networks (if not passed, uses the dataset default choice)'
)
parser.add_argument(
    '-b',
    '--batch',
    type=int,
    nargs='?',
    help='the batch size used during training (if not passed, uses the dataset default choice)'
)
parser.add_argument(
    '-t',
    '--threshold',
    type=float,
    nargs='?',
    help='the penalty threshold used during training (if not passed, uses the dataset default choice)'
)
parser.add_argument(
    '-a',
    '--alpha',
    type=float,
    nargs='?',
    help='the alpha value used for the penalty constraint (if not passed, uses automatic tuning)'
)
parser.add_argument(
    '-p',
    '--wandb-project',
    type=str,
    nargs='?',
    help='the name of the Weights & Biases project for logging, or None for no logging'
)
parser.add_argument(
    '-e',
    '--extensions',
    type=str,
    nargs='*',
    choices=['csv', 'tex'],
    default=['csv'],
    help='the extensions of the files to save'
)

# parse arguments, build experiments, then export the results
args = parser.parse_args().__dict__
print("Starting experiment 'results'...")
for k, v in args.items():
    print('  >', k, '-->', v)
print()
args['datasets'] = [datasets[k] for k in args['datasets']]
args['metrics'] = {k: v for k, v in [metrics(m) for m in args['metrics']]}
LearningExperiment.outputs(**args)