"""
This script is the main file that calls all the other
scripts to run the ml project.
"""

import os
import joblib
import pandas as pd
import src.config as sc

from sklearn import metrics
from src.model_dispatcher import model


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

    clf = model

    """fit model on the training data"""
    clf.fit(x_train, y_train)

    """As target variable is skewed we will need predicted probabilities to calculate AUC score"""
    y_pred = clf.predict_proba(x_valid)[:,1]

    """find accuracy as distribution of all target variables in similar"""
    auc = metrics.roc_auc_score(y_valid, y_pred)
    print(f"Fold number :{fold}, AUC score : {auc}")

    """Save Model"""
    joblib.dump(clf, os.path.join(sc.OUTPUT_FILE, f'dt_{fold}.bin'))


if __name__ == '__main__':
    df = pd.read_csv(sc.TRAINING_FILE)
    run_output(df)
