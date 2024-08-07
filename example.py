import argparse
import logging

from experiments import AnalysisExperiment
from items.datasets import Polynomial, NonLinear, Communities, Adult, Census

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
    if key == 'linear':
        return Polynomial(degree_x=1, degree_y=1, noise=noise, seed=0)
    elif key == 'x_square':
        return Polynomial(degree_x=2, degree_y=1, noise=noise, seed=0)
    elif key == 'x_cubic':
        return Polynomial(degree_x=3, degree_y=1, noise=noise, seed=0)
    elif key == 'y_square':
        return Polynomial(degree_x=1, degree_y=2, noise=noise, seed=0)
    elif key == 'y_cubic':
        return Polynomial(degree_x=1, degree_y=3, noise=noise, seed=0)
    elif key == 'circle':
        return Polynomial(degree_x=2, degree_y=2, noise=noise, seed=0)
    elif key == 'sign':
        return NonLinear(fn='sign', noise=noise, seed=0)
    elif key == 'relu':
        return NonLinear(fn='relu', noise=noise, seed=0)
    elif key == 'sin':
        return NonLinear(fn='sin', noise=noise, seed=0)
    elif key == 'tanh':
        return NonLinear(fn='tanh', noise=noise, seed=0)
    else:
        raise KeyError(f"Invalid key '{key}' for dataset")


# build argument parser
parser = argparse.ArgumentParser(description='Plot the example figure')
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
    default='circle-0.1',
    help='the dataset on which to run the experiment'
)
parser.add_argument(
    '-da',
    '--degree_a',
    type=int,
    default=2,
    help='the degree for the a variable'
)
parser.add_argument(
    '-db',
    '--degree_b',
    type=int,
    default=2,
    help='the degree for the b variable'
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
print("Building example plot...")
for k, v in args.items():
    print('  >', k, '-->', v)
print()
args['dataset'] = dataset(key=args['dataset'])
AnalysisExperiment.example(**args)
