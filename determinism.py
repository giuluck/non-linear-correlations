import argparse
import logging

from experiments import CorrelationExperiment

log = logging.getLogger("lightning_fabric")
log.propagate = False
log.setLevel(logging.ERROR)

# build argument parser
parser = argparse.ArgumentParser(description='Test determinism of HGR indicators')
parser.add_argument(
    '-f',
    '--folder',
    type=str,
    default='results',
    help='the path where to search and store the results and the exports'
)
parser.add_argument(
    '-d',
    '--dataset',
    type=str,
    default='square_sin',
    help='the dataset on which to run the experiment'
)
parser.add_argument(
    '-n',
    '--noises',
    type=float,
    nargs='+',
    default=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.2, 2.4, 2.6, 2.8, 3.0],
    help='the noise values used in the experiments'
)
parser.add_argument(
    '-s',
    '--seeds',
    type=int,
    nargs='+',
    default=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
             16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29],
    help='the number of tests per experiment'
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
print("Starting experiment 'determinism'...")
for k, v in args.items():
    print('  >', k, '-->', v)
print()
CorrelationExperiment.determinism(**args)
