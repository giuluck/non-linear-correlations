import argparse
import logging
import re

from experiments import CorrelationExperiment
from items.datasets import Synthetic
from items.indicators import DoubleKernelHGR, DensityHGR, ChiSquare, RandomizedDependenceCoefficient, SingleKernelHGR, \
    AdversarialHGR

# noinspection DuplicatedCode
log = logging.getLogger("lightning_fabric")
log.propagate = False
log.setLevel(logging.ERROR)


# function to retrieve the valid indicator
def indicators(key):
    if key == 'nn':
        return 'HGR-NN', AdversarialHGR()
    elif key == 'kde':
        return 'HGR-KDE', DensityHGR()
    elif key == 'chi':
        return 'CHI^2', ChiSquare()
    elif key == 'rdc':
        return 'RDC', RandomizedDependenceCoefficient()
    elif key == 'prs':
        return 'PEARS', DoubleKernelHGR(degree_a=1, degree_b=1)
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
        raise KeyError(f"Invalid key '{key}' for indicator")


# build argument parser
parser = argparse.ArgumentParser(description='Test multiple HGR indicators on multiple datasets')
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
    default=['circle', 'square_sin'],
    choices=list(Synthetic.FUNCTIONS.keys()),
    help='the datasets on which to run the experiment'
)
parser.add_argument(
    '-i',
    '--indicators',
    type=str,
    nargs='+',
    default=['kb', 'sk', 'nn', 'kde', 'rdc', 'prs'],
    help='the indicator used to compute the correlations'
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
    '-n',
    '--noises',
    type=float,
    nargs='+',
    default=[0.0, 1.0, 3.0],
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
parser.add_argument(
    '-t',
    '--save-time',
    type=int,
    default=60,
    help='the number of seconds after which to store the computed results'
)

# parse arguments, build experiments, then export the results
args = parser.parse_args().__dict__
print("Starting experiment 'scalability'...")
for k, v in args.items():
    print('  >', k, '-->', v)
print()
args['indicators'] = {k: v for k, v in [indicators(mt) for mt in args['indicators']]}
CorrelationExperiment.scalability(**args)
