import argparse
import logging
import os
import warnings

from experiments import LearningExperiment
from items.datasets import Communities, Adult, Census

os.environ['WANDB_SILENT'] = 'true'
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

# list the default units to be used in case no units are passed
units = [[32], [256], [32, 32], [256, 256], [32, 32, 32], [256, 256, 256]]

# build argument parser
parser = argparse.ArgumentParser(description='Selects the best number of units for unconstrained neural networks')
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
    '-b',
    '--batches',
    type=int,
    nargs='*',
    default=[512, 4096, -1],
    help='the number of batches used during training (e.g., 1 for full batch)'
)
parser.add_argument(
    '-u',
    '--units',
    nargs='*',
    type=int,
    action='append',
    help='the hidden units used in the experiment'
)
parser.add_argument(
    '-s',
    '--steps',
    type=int,
    default=1000,
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
    default=['png'],
    help='the extensions of the files to save'
)
parser.add_argument(
    '--plot',
    action='store_true',
    help='whether to plot the results'
)

# parse arguments, build experiments, then export the results
args = parser.parse_args().__dict__
args['units'] = units if args['units'] is None else args['units']
print("Starting experiment 'calibration'...")
for k, v in args.items():
    print('  >', k, '-->', v)
print()
args['datasets'] = [datasets[k] for k in args['datasets']]
LearningExperiment.calibration(**args)
