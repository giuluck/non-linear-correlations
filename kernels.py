import argparse
import logging
import re

from experiments import CorrelationExperiment
from items.datasets import Deterministic
from items.hgr import AdversarialHGR, DoubleKernelHGR, SingleKernelHGR

log = logging.getLogger("lightning_fabric")
log.propagate = False
log.setLevel(logging.ERROR)


# function to retrieve the valid metric
def metrics(key):
    if key == 'nn':
        return 'HGR-NN', AdversarialHGR()
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
        raise KeyError(f"Invalid key '{key}' for metric")


# build argument parser
parser = argparse.ArgumentParser(description='Inspect the HGR kernels on a given dataset')
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
    choices=list(Deterministic.FUNCTIONS.keys()),
    default=['circle'],
    help='the dataset on which to run the experiment'
)
parser.add_argument(
    '-m',
    '--metrics',
    type=str,
    nargs='*',
    default=['kb', 'kb-2', 'nn'],
    help='the metric used to compute the correlations'
)
parser.add_argument(
    '-n',
    '--noises',
    type=float,
    nargs='+',
    default=[1.0],
    help='the noise values used in the experiments'
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
print("Starting experiment 'kernels'...")
for k, v in args.items():
    print('  >', k, '-->', v)
print()
args['metrics'] = {k: v for k, v in [metrics(key=mt) for mt in args['metrics']]}
CorrelationExperiment.kernels(**args)
