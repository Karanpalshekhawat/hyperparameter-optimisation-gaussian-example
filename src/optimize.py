"""
This script have the optimization function.
it initilises the model with the chosen parameter,
runs cross validation and compute accuracy and
returns the negative of accuracy
"""

import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn import model_selection, metrics


def opt_func(params, param_names, x, y):
    params = dict(zip(param_names, params))
    model = RandomForestClassifier(
        **params)  # it is the way to pass a dictionary if your dictionary have same keys as arguments for module

    kf = model_selection.StratifiedKFold(n_splits=5)

    # accuracy list to store accuracy values for different folds
    accuracy = []

    for idx in kf.split(X=x, y=y):
        train_idx, test_idx = idx[0], idx[1]
        x_train = x[train_idx]
        y_train = y[train_idx]
        x_test = x[test_idx]
        y_test = y[test_idx]

        model.fit(x_train, y_train)
        preds = model.predict(x_test)

        accuracy.append(metrics.accuracy_score(y_test, preds))

    return -1 * np.mean(accuracy)
