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

    print(clf.best_score_)
    print(clf.best_estimator_.get_params())

    return clf


if __name__ == '__main__':
    df = pd.read_csv(sc.TRAINING_FILE)
    run_output(df)
