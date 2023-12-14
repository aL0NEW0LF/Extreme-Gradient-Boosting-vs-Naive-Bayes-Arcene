import os
import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_selection import chi2
import eexgboost as xgb
import exgboost as exgb

class SquaredErrorObjective():
    def loss(self, y, pred): return np.mean((y - pred)**2)
    def gradient(self, y, pred): return pred - y
    def hessian(self, y, pred): return np.ones(len(y))

if __name__ == '__main__':
    arcene_train = pd.read_csv("./data/ARCENE/arcene_train.data", sep=" ", header=None).drop([10000], axis=1)
    arcene_train_labels = pd.read_csv("./data/ARCENE/arcene_train.labels", sep=" ", header=None)
    arcene_valid = pd.read_csv("./data/ARCENE/arcene_valid.data", sep=" ", header=None).drop([10000], axis=1)
    arcene_valid_labels = pd.read_csv("./data/arcene_valid.labels", sep=" ", header=None)

    params = {
        'learning_rate': 0.1,
        'max_depth': 5,
        'subsample': 0.8,
        'reg_lambda': 1.5,
        'gamma': 0.0,
        'min_child_weight': 25,
        'base_score': 0.0,
        'tree_method': 'exact',
    }
    num_boost_round = 50

    xgboostclass = exgb.XGBoostModel(params, random_seed=42)
    xgboostclass.fit(arcene_train, arcene_train_labels, SquaredErrorObjective(), num_boost_round)
    pred_scratch = xgboostclass.predict(arcene_valid)
    print(f'scratch score: {SquaredErrorObjective().loss(arcene_valid_labels, pred_scratch)}')
