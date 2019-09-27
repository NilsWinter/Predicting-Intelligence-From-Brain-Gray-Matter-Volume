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
from photonai.base.PhotonBase import OutputSettings, Hyperpipe, PipelineElement
from photonai.optimization.Hyperparameters import Categorical, FloatRange


def construct_hyperpipe(name, project_folder, cache_dir: str = '../../cache'):
    # cv
    outer_cv = StratifiedKFoldRegression(n_splits=10, shuffle=True, random_state=3)
    inner_cv = StratifiedKFoldRegression(n_splits=3, shuffle=True, random_state=4)

    # define output
    output = OutputSettings(
        save_predictions='best',
        save_feature_importances='None',
        project_folder=project_folder)

    # define hyperpipe
    pipe = Hyperpipe(name=name,
                     optimizer='sk_opt',
                     optimizer_params={'num_iterations': 50, 'base_estimator': 'GP'},
                     metrics=['mean_squared_error', 'mean_absolute_error', 'variance_explained'],
                     best_config_metric='mean_squared_error',
                     outer_cv=outer_cv,
                     inner_cv=inner_cv,
                     eval_final_performance=True,
                     verbosity=1,
                     output_settings=output)

    # add elements to hyperpipe
    # add confounder removal first
    pipe += PipelineElement('ConfounderRemoval', {}, standardize_covariates=True, cache_dir=cache_dir,
                            test_disabled=False)
    pipe += PipelineElement('PhotonPCA', {}, n_components=None,  test_disabled=False,
                            logs=cache_dir)

    pipe += PipelineElement('LinearSVR', {
        'C': FloatRange(0.000001, 1, range_type='linspace'),
        'epsilon': FloatRange(0.01, 3, range_type='linspace')})
    return pipe