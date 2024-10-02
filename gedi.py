import argparse
import logging
import os
import re
import warnings

from experiments import LearningExperiment
from items.datasets import Communities, Adult, Census
from items.indicators import DoubleKernelHGR, SingleKernelHGR, AdversarialHGR, DensityHGR

# noinspection DuplicatedCode
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings("ignore", ".*does not have many workers.*")
for name in ["lightning_fabric", "pytorch_lightning.utilities.rank_zero", "pytorch_lightning.accelerators.cuda"]:
    log = logging.getLogger(name)
    log.propagate = False
    log.setLevel(logging.ERROR)


# list all the valid datasets
datasets = dict(
    communities=Communities(),
    adult=Adult(),
    census=Census()
)

# build argument parser
parser = argparse.ArgumentParser(description='Train multiple models using the GeDI indicator as constraint')
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
    default=['communities', 'adult', 'census'],
    help='the datasets on which to run the experiment'
)
parser.add_argument(
    '-c',
    '--constraint',
    type=str,
    default='both',
    choices=['fine', 'coarse', 'both'],
    help='the type of constraint'
)
parser.add_argument(
    '-l',
    '--learners',
    type=str,
    nargs='*',
    default=['lm', 'rf', 'gb', 'nn'],
    choices=['lm', 'rf', 'gb', 'nn'],
    help='the learning models'
)
parser.add_argument(
    '-m',
    '--methods',
    type=str,
    nargs='*',
    default=['mt', 'ld'],
    choices=['mt', 'ld'],
    help='the constraint enforcement methods'
)
parser.add_argument(
    '-k',
    '--folds',
    type=int,
    default=5,
    help='the number of folds to be used for cross-validation'
)
parser.add_argument(
    '-e',
    '--extensions',
    type=str,
    nargs='*',
    default=['png', 'csv'],
    choices=['png', 'pdf', 'csv', 'tex'],
    help='the extensions of the files to save'
)
parser.add_argument(
    '--plot',
    action='store_true',
    help='whether to plot the results'
)

# parse arguments, build experiments, then export the results
args = parser.parse_args().__dict__
print("Starting experiment 'gedi'...")
for k, v in args.items():
    print('  >', k, '-->', v)
print()
args['datasets'] = [datasets[k] for k in args['datasets']]
LearningExperiment.gedi(**args)
