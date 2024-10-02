import argparse
import logging

from experiments import CorrelationExperiment
from items.datasets import Synthetic, Communities, Adult, Census

log = logging.getLogger("lightning_fabric")
log.propagate = False
log.setLevel(logging.ERROR)

# list all the valid datasets
datasets = dict(
    adult=Adult(),
    census=Census(),
    communities=Communities(),
    **{name: Synthetic(name=name) for name in Synthetic.FUNCTIONS.keys()}
)

# build argument parser
parser = argparse.ArgumentParser(description='Test the Kernel-based HGR on a given dataset')
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
    help='the dataset on which to run the experiment'
)
parser.add_argument(
    '-da',
    '--degrees_a',
    type=int,
    nargs='+',
    default=[1, 2, 3, 4, 5, 6, 7],
    help='the degrees for the a variable'
)
parser.add_argument(
    '-db',
    '--degrees_b',
    type=int,
    nargs='+',
    default=[1, 2, 3, 4, 5, 6, 7],
    help='the degrees for the b variable'
)
parser.add_argument(
    '-m',
    '--vmin',
    type=float,
    nargs='?',
    help='the min value used in the color bar (or None if empty)'
)
parser.add_argument(
    '-M',
    '--vmax',
    type=float,
    nargs='?',
    help='the max value used in the color bar (or None if empty)'
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
print("Starting experiment 'monotonicity'...")
for k, v in args.items():
    print('  >', k, '-->', v)
print()
args['datasets'] = [datasets[name] for name in args['datasets']]
CorrelationExperiment.monotonicity(**args)
