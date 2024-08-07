import argparse
import logging

from experiments import AnalysisExperiment
from items.datasets import Communities, Adult, Census

log = logging.getLogger("lightning_fabric")
log.propagate = False
log.setLevel(logging.ERROR)

# list all the valid datasets
datasets = dict(
    communities=Communities(),
    adult=Adult(),
    census=Census()
)

# build argument parser
parser = argparse.ArgumentParser(description='Leverages HGR to compute the importance of features')
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
    '-on',
    '--on',
    type=str,
    choices=['protected', 'target', 'both'],
    default='both',
    help='the variable with respect to which the importance is computed'
)
parser.add_argument(
    '-t',
    '--top',
    type=int,
    default=10,
    help='the top features to show'
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
print("Starting experiment 'importance'...")
for k, v in args.items():
    print('  >', k, '-->', v)
print()
args['datasets'] = [datasets[dataset] for dataset in args['datasets']]
AnalysisExperiment.importance(**args)
