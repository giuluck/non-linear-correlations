import argparse
import logging

from experiments import ConstraintExperiment
from items.datasets import Deterministic, Communities, Adult, Census

log = logging.getLogger("lightning_fabric")
log.propagate = False
log.setLevel(logging.ERROR)


# function to retrieve the valid dataset
def dataset(key):
    if key == 'communities':
        return Communities()
    elif key == 'adult':
        return Adult()
    elif key == 'census':
        return Census()
    noise = 0.0
    if '-' in key:
        key, noise = key.split('-')
        noise = float(noise)
    if key in Deterministic.FUNCTIONS:
        return Deterministic(name=key, noise=noise, seed=0)
    raise KeyError(f"Invalid key '{key}' for dataset")


# build argument parser
parser = argparse.ArgumentParser(description='Uses the declarative formulations of GeDI to project the training data')
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
    default=['communities', 'adult', 'census'],
    help='the datasets on which to run the experiment'
)
parser.add_argument(
    '-k',
    '--degrees',
    type=int,
    nargs='+',
    default=[1, 2, 3, 4, 5],
    help='the GeDI degrees to be tested'
)
parser.add_argument(
    '-b',
    '--bins',
    type=int,
    nargs='+',
    default=[2, 3, 5, 10],
    help='the number of bins to be tested for the Binned DIDI metric'
)
parser.add_argument(
    '-t',
    '--threshold',
    type=float,
    default=0.2,
    help='the threshold up to which to exclude the feature'
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
print("Starting experiment 'projections'...")
for k, v in args.items():
    print('  >', k, '-->', v)
print()
args['datasets'] = [dataset(key=key) for key in args['datasets']]
ConstraintExperiment.projections(**args)
