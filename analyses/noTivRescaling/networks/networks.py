"""
===========================================================
Project: IQ-Prediction Frankfurt
===========================================================
Description
-----------
This scripts implements single network analysis using NoTivRescaling

Version
-------
Created:        28-01-2019
Last updated:   17-03-2019

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


def run_network(network_index):
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

    analysis_name = network_names[network_index] + '_second_test_run'
    data_folder = '/scratch/tmp/wintern/iq_frankfurt/'
    project_folder = '/scratch/tmp/wintern/iq_frankfurt/results/noTivRescaling/networks/' + network_names[network_index]
    cache_dir = '/scratch/tmp/wintern/cache'

    # get data
    data = IQData(data_folder=data_folder, tiv_rescaled=False)
    covariates = np.asarray([data.age, data.gender, data.handedness]).T
    data.load_single_networks(use_cached=False)
    X = data.networks[network_index]
    y = data.fsiq
    del data

    # run analysis
    pipe = construct_hyperpipe(analysis_name, project_folder, cache_dir)
    pipe.fit(X, y, **{'covariates': covariates})
    os.remove(pipe.output_settings.pretrained_model_filename)

run_network(sys.argv[1])
