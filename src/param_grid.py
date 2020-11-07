"""
This script just define the parameter
grid for different hyper parameters
"""
import numpy as np

parameter_grid = {
    'n_estimators': [100, 200, 300, 400, 500],
    'max_depth': [1, 2, 5, 7, 11, 15],
    'criterion': ['gini', 'entropy']
}
