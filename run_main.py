"""
This script is the main file that calls all the other
scripts to run the ml project.
"""

import pandas as pd
import src.config as sc

from functools import partial
from skopt import gp_minimize
from src.param_grid import param_space, param_names
from src.optimize import opt_func


def run_output(df):
    """
    Structure, train and save the model
    for given fold number.

    Args:
        df (pd.DataFrame): training dataset

    Returns:

    """
    features = [i for i in df.columns if i != 'price_range']
    X = df[features].values
    y = df['price_range'].values

    optimization_function = partial(opt_func, param_names=param_names, x=X, y=y)

    result = gp_minimize(optimization_function, dimensions=param_space, n_calls=15, n_random_starts=10, verbose=10)
    best_parameters = dict(zip(param_names, result.x))

    return best_parameters


if __name__ == '__main__':
    df = pd.read_csv(sc.TRAINING_FILE)
    """
    One very important thing is that we are trying to minimise something 
    but we cannot minimise accuracy as we should maximise it, so we multiply
    accuracy by -1 and then minimise it. This way we are minimising accuracy
    but in fact we are maximising accuracy.
    """
    print(run_output(df))
