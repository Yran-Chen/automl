from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from bayes_opt import BayesianOptimization
import numpy as np


def rf_cv(n_estimators, min_samples_split, max_features, max_depth):
    val = cross_val_score(
        RandomForestClassifier(n_estimators=int(n_estimators),
            min_samples_split=int(min_samples_split),
            max_features=min(max_features, 0.999), # float
            max_depth=int(max_depth),
            random_state=2
        ),
        x, y, scoring='roc_auc', cv=5
    ).mean()
    return val


if __name__ == '__main__':
    x, y = make_classification(n_samples=2000, n_features=15, n_classes=2)
    # print (x,y)
    # rf = RandomForestClassifier()
    # print(np.mean(cross_val_score(rf, x, y, cv=20, scoring='roc_auc')))

    rf_bo = BayesianOptimization(
        rf_cv,
        {'n_estimators': (10, 250),
        'min_samples_split': (2, 25),
        'max_features': (0.1, 0.999),
        'max_depth': (5, 15)}
    )
    rf_bo.maximize()