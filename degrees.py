import argparse
import logging

from experiments import FigureExperiment

log = logging.getLogger("lightning_fabric")
log.propagate = False
log.setLevel(logging.ERROR)

# build argument parser
parser = argparse.ArgumentParser(description='Plot the GeDI degree enforcement example')
parser.add_argument(
    '-f',
    '--folder',
    type=str,
    default='results',
    help='the path where to search and store the results and the exports'
)
parser.add_argument(
    '--coarse',
    action='store_true',
    help='whether to use the coarse-grained or fine-grained GeDI formulation'
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
print("Starting experiment 'degrees'...")
for k, v in args.items():
    print('  >', k, '-->', v)
print()
FigureExperiment.degrees(**args)
