"""
===========================================================
Project: Predicting Intelligence from Brain Gray Matter
===========================================================
Description
-----------
Base script for every analysis within this project. Hyperpipe structure is defined here and used in the analysis
scripts.

PHOTON Branch
-------------
project/IQ_prediction_280119

Version
-------
Created:        28-01-2019
Last updated:   27-09-2019

Author
------
Nils R. Winter
nils.r.winter@gmail.com
Translationale Psychiatrie
Universitaetsklinikum Muenster
"""

from photonai.validation.cross_validation import StratifiedKFoldRegression
from sklearn.dummy import DummyRegressor
from DemoFiles.Nils.iq_prediction.data.data import IQData
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
import pandas as pd


data_folder = '../../'
data = IQData(data_folder=data_folder)
y = np.asarray(data.fsiq)
X = np.ones((y.shape[0], 2))
outer_cv = StratifiedKFoldRegression(n_splits=10, shuffle=True, random_state=3)

results = {'mae': list(), 'mse': list(), 'rmse': list()}

for train, test in outer_cv.split(X, y):
    X_train = X[train]
    X_test = X[test]
    y_train = y[train]
    y_test = y[test]

    dummy = DummyRegressor(strategy='mean')
    dummy.fit(X_train, y_train)
    y_test_pred = dummy.predict(X_test)

    results['mae'].append(mean_absolute_error(y_test, y_test_pred))
    results['mse'].append(mean_squared_error(y_test, y_test_pred))
    results['rmse'].append(np.sqrt(mean_squared_error(y_test, y_test_pred)))

results['mean_mae'] = np.mean(results['mae'])
results['mean_mse'] = np.mean(results['mse'])
results['mean_rmse'] = np.mean(results['rmse'])
results['std_mae'] = np.std(results['mae'])
results['std_mse'] = np.std(results['mse'])
results['std_rmse'] = np.std(results['rmse'])


df = pd.DataFrame(results)
df.to_csv('dummy_results.csv')