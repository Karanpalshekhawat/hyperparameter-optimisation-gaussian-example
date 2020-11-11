"""
This script is the main file that calls all the other
scripts to run the ml project.
"""

import pandas as pd
import src.config as sc

from sklearn import model_selection
from src.model_dispatcher import model
from src.param_grid import parameter_grid


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

    """
    Create a parameter space using scikit-optimize library (skopt)
    """

    clf = model_selection.GridSearchCV(
        estimator=model,
        param_grid=parameter_grid,
        scoring='accuracy',
        verbose=10,  # higher value of it just means that lot of information will be printed
        n_jobs=1,
        cv=5
    )

    """fit model on the training data"""
    clf.fit(X, y)

    print(f'Best score is : {clf.best_score_}')
    best_parameters = clf.best_estimator_.get_params()
    print('Best Parameters Set:')
    for prm in parameter_grid.keys():
        print(f'\t {prm} : {best_parameters[prm]}')

    return clf


if __name__ == '__main__':
    df = pd.read_csv(sc.TRAINING_FILE)
    """
    One very important thing is that we are trying to minimise something 
    but we cannot minimise accuracy as we should maximise it, so we multiply
    accuracy by -1 and then minimise it. This way we are minimising accuracy
    but in fact we are maximising accuracy.
    """
    run_output(df)
