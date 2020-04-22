"""
===========================================================
Project: IQ-Prediction Frankfurt
===========================================================
Description
-----------
Permutation script for schaefer analysis

Version
-------
Created:        28-01-2020
Last updated:   28-01-2020

Author
------
Nils R. Winter
nils.r.winter@gmail.com
Translationale Psychiatrie
Universitaetsklinikum Muenster
"""
import sys
sys.path.append('/scratch/tmp/wintern/iq_frankfurt/photonai')
sys.path.append('/scratch/tmp/wintern/iq_frankfurt/')
from analyses.analysis_base import construct_hyperpipe_schaefer
from data.data import IQData
import numpy as np
import os


def run_perm_test(row):
    row_array = row.split(';')
    perm_ind = row_array[0]
    perm_y = row_array[1]
    perm_y = np.asarray(perm_y.split(',')).astype(int)

    analysis_name = 'schaefer100_perm_' + str(perm_ind)
    data_folder = '/scratch/tmp/wintern/iq_frankfurt/'
    project_folder = '/scratch/tmp/wintern/iq_frankfurt/results/TivRescaling/perm/schaefer100/'
    cache_dir = '/scratch/tmp/wintern/cache'

    # get data
    data = IQData(data_folder=data_folder)
    covariates = np.asarray([data.age, data.gender, data.handedness]).T
    data.load_schaefer100(use_cached=True)

    # run analysis
    pipe = construct_hyperpipe_schaefer(analysis_name, project_folder, cache_dir)
    pipe.groups = data.fsiq
    pipe.fit(data.schaefer100, perm_y, **{'covariates': covariates})
    os.remove(pipe.output_settings.pretrained_model_filename)

run_perm_test(sys.argv[1])
