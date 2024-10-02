import argparse
import logging

from experiments import CorrelationExperiment
from items.datasets import Synthetic

log = logging.getLogger("lightning_fabric")
log.propagate = False
log.setLevel(logging.ERROR)

# build argument parser
parser = argparse.ArgumentParser(description='Compare solving times using Least-Square vs Global Optimization')
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
    default=['circle', 'square_sin'],
    choices=list(Synthetic.FUNCTIONS.keys()),
    help='the dataset on which to run the experiment'
)
parser.add_argument(
    '-n',
    '--noise',
    type=float,
    default=0.5,
    help='the noise values used in the experiments'
)
parser.add_argument(
    '-s',
    '--seeds',
    type=int,
    nargs='+',
    default=[0, 1, 2, 3, 4],
    help='the number of tests per experiment'
)
parser.add_argument(
    '-z',
    '--sizes',
    type=int,
    nargs='+',
    default=[11, 51, 101, 501, 1001, 5001, 10001, 50001, 100001, 500001, 1000001],
    help='the size of the generated datasets to test'
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
print("Starting experiment 'lstsq'...")
for k, v in args.items():
    print('  >', k, '-->', v)
print()
CorrelationExperiment.lstsq(**args)
