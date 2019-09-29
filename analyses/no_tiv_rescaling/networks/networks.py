"""
===========================================================
Project: Predicting Intelligence from Brain Gray Matter
===========================================================
Description
-----------
Permutation script for network analysis

Version
-------
Created:        02-02-2019
Last updated:   29-09-2019

Author
------
Nils R. Winter
nils.r.winter@gmail.com
Translationale Psychiatrie
Universitaetsklinikum Muenster
"""
import sys
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

    analysis_name = network_names[network_index] + "_no_tiv_rescaling"
    project_folder = '.'

    # get data
    data = IQData(tiv_rescaled=False)
    covariates = np.asarray([data.age, data.gender, data.handedness]).T
    data.load_single_networks(use_cached=False)
    X = data.networks[network_index]
    y = data.fsiq
    del data

    # run analysis
    pipe = construct_hyperpipe(analysis_name, project_folder)
    pipe.fit(X, y, **{'covariates': covariates})
    os.remove(pipe.output_settings.pretrained_model_filename)

run_network(sys.argv[1])
