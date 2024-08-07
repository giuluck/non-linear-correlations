import argparse
import logging

from experiments import AnalysisExperiment
from items.datasets import Communities, Adult, Census, Deterministic

log = logging.getLogger("lightning_fabric")
log.propagate = False
log.setLevel(logging.ERROR)

# list all the valid datasets
datasets = dict(
    communities=Communities(),
    adult=Adult(),
    census=Census(),
    **{name: Deterministic(name=name) for name in Deterministic.FUNCTIONS.keys()}
)

# build argument parser
parser = argparse.ArgumentParser(description='Leverages HGR to assume the direction of the causal link')
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
print("Starting experiment 'causality'...")
for k, v in args.items():
    print('  >', k, '-->', v)
print()
args['datasets'] = [datasets[dataset] for dataset in args['datasets']]
AnalysisExperiment.causality(**args)
