"""
===========================================================
Project: IQ-Prediction Frankfurt
===========================================================
Description
-----------
Permutation script for whole brain analysis with NoTivRescaling

Version
-------
Created:        28-01-2019
Last updated:   15-03-2019

Author
------
Nils R. Winter
nils.r.winter@gmail.com
Translationale Psychiatrie
Universitaetsklinikum Muenster
"""
import os
import sys
import numpy as np
from analyses.analysis_base import construct_hyperpipe
from data.data import IQData


def run_perm_test(row):
    row_array = row.split(';')
    perm_ind = row_array[0]
    perm_y = row_array[1]
    perm_y = np.asarray(perm_y.split(',')).astype(int)

    analysis_name = 'whole_brain_noTiv_perm_' + str(perm_ind)
    data_folder = '/scratch/tmp/wintern/iq_frankfurt/'
    project_folder = './perm/'

    # get data
    data = IQData(data_folder=data_folder, tiv_rescaled=False)
    covariates = np.asarray([data.age, data.gender, data.handedness]).T
    data.load_whole_brain(use_cached=True)

    # run analysis
    pipe = construct_hyperpipe(analysis_name, project_folder)
    pipe.groups = data.fsiq
    pipe.fit(data.whole_brain, perm_y, **{'covariates': covariates})
    os.remove(pipe.output_settings.pretrained_model_filename)

run_perm_test(sys.argv[1])
