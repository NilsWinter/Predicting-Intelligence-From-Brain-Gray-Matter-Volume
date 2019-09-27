"""
===========================================================
Project: IQ-Prediction Frankfurt
===========================================================
Description
-----------
Permutation script for network analysis

Version
-------
Created:        02-02-2019
Last updated:   02-02-2019

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
from analyses.analysis_base import construct_hyperpipe
from data.data import IQData
import numpy as np
import os


def run_perm_test(row, network_index):
    network_index = int(network_index)
    network_names = {0: "visual",
                     1: "somatomotor",
                     2: "dorsal_attention",
                     3: "ventral_attention",
                     4: "limbic",
                     5: "fronto_parietal",
                     6: "default_mode",
                     7: "subcortical",
                     8: "cerebellum"}

    row_array = row.split(';')
    perm_ind = row_array[0]
    perm_y = row_array[1]
    perm_y = np.asarray(perm_y.split(',')).astype(int)

    analysis_name = network_names[network_index] + '_perm_' + str(perm_ind)
    data_folder = '/scratch/tmp/wintern/iq_frankfurt/'
    project_folder = '/scratch/tmp/wintern/iq_frankfurt/results/perm/networks/'
    cache_dir = '/scratch/tmp/wintern/cache'

    # get data
    data = IQData(data_folder=data_folder)
    covariates = np.asarray([data.age, data.gender, data.handedness]).T
    data.load_single_networks(use_cached=True)
    X = data.networks[network_index]
    y = data.fsiq
    del data

    # run analysis
    pipe = construct_hyperpipe(analysis_name, project_folder, cache_dir)
    pipe.groups = y
    pipe.fit(X, perm_y, **{'covariates': covariates})
    os.remove(pipe.output_settings.pretrained_model_filename)

run_perm_test(sys.argv[1], sys.argv[2])
