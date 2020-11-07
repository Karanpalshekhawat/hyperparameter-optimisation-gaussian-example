"""
This script just define the parameter
grid for different hyper parameters
"""
import numpy as np

parameter_grid = {
    'n_estimators': np.arange(100, 1500, 100),
    'max_depth': np.arange(1, 31),
    'criterion': ['gini', 'entropy']
}
