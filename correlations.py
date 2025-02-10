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
    choices=list(Synthetic.FUNCTIONS.keys()),
    default=list(Synthetic.FUNCTIONS.keys()),
    help='the datasets on which to run the experiment'
)
parser.add_argument(
    '-i',
    '--indicators',
    type=str,
    nargs='*',
    default=['kb', 'sk', 'nn', 'kde', 'rdc', 'prs'],
    help='the indicator used to compute the correlations'
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
    '-ns',
    '--noise-seeds',
    type=int,
    nargs='+',
    default=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
    help='the number of dataset variants per experiment'
)
parser.add_argument(
    '-as',
    '--algorithm-seeds',
    type=int,
    nargs='+',
    default=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
    help='the number of tests per experiment'
)
parser.add_argument(
    '--test',
    action='store_true',
    help='whether to compute the correlations on test data'
)
parser.add_argument(
    '-c',
    '--columns',
    type=int,
    default=3,
    help='the number of columns in the final plot'
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
print("Starting experiment 'correlations'...")
for k, v in args.items():
    print('  >', k, '-->', v)
print()
args['indicators'] = {k: v for k, v in [indicators(mt) for mt in args['indicators']]}
CorrelationExperiment.correlations(**args)
