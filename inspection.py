import argparse

from experiments import AnalysisExperiment, CorrelationExperiment, LearningExperiment, ConstraintExperiment

# list all the available experiment files
EXPERIMENTS = {
    'analysis': AnalysisExperiment,
    'constraint': ConstraintExperiment,
    'correlation': CorrelationExperiment,
    'learning': LearningExperiment
}

# build argument parser
parser = argparse.ArgumentParser(description='Inspects the signature of the experiments')
parser.add_argument(
    '-f',
    '--folder',
    type=str,
    default='results',
    help='the path where to search and store the results and the exports'
)
parser.add_argument(
    '-x',
    '--experiments',
    type=str,
    nargs='+',
    default=EXPERIMENTS.keys(),
    choices=EXPERIMENTS.keys(),
    help='the name of the experiment (or list of such) to clear'
)
parser.add_argument(
    '-e',
    '--export',
    type=str,
    nargs='*',
    default=['json'],
    choices=['csv', 'json'],
    help='stores an export file with the signatures of the experiments'
)
parser.add_argument(
    '--show',
    action='store_true',
    help='prints the signatures of the experiments on screen'
)

# parse arguments and run
args = parser.parse_args().__dict__
print('Starting experiments inspection procedure...')
for k, v in args.items():
    print('  >', k, '-->', v)
print()
for experiment in args.pop('experiments'):
    print(f'{experiment.upper()} EXPERIMENT:', end='')
    EXPERIMENTS[experiment].inspection(**args)
    print()
