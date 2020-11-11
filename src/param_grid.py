"""
This script just define the parameter space
grid for different hyper parameters
"""
from skopt import space

param_space = [
    space.Integer(3, 15, name='max_depth'),
    space.Integer(100,1500, name='n_estimators'),
    space.Categorical(['gini', 'entropy'], name='criterion'),
    space.Real(0.01, 1, prior='uniform', name='max_features')
]

param_names = [
    'max_depth',
    'n_estimators',
    'criterion',
    'max_features'
]