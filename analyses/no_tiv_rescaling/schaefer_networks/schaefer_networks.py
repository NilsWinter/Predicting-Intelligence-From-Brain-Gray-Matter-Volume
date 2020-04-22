"""
===========================================================
Project: IQ-Prediction Frankfurt
===========================================================
Description
-----------
This scripts implements schaefer network analysis with NoTivRescaling
(This is the data that hasn't been rescaled by Kirsten)

Version
-------
Created:        28-01-2019
Last updated:   18-02-2019

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


networks = ['visual', 'somatomotor', 'dorsal_attention', 'ventral_attention', 'limbic',
            'default_mode', 'fronto_parietal']

for network in networks:
    analysis_name = 'schaefer_{}_no_tiv'.format(network)
    data_folder = '/scratch/tmp/wintern/iq_frankfurt/'
    project_folder = '/scratch/tmp/wintern/iq_frankfurt/results/noTivRescaling/schaefer_{}'.format(network)
    os.makedirs(project_folder, exist_ok=True)
    cache_dir = '/scratch/tmp/wintern/cache'

    # get data
    data = IQData(data_folder=data_folder, tiv_rescaled=False)
    covariates = np.asarray([data.age, data.gender, data.handedness]).T
    data.load_schaefer_network(network, use_cached=False)

    # run analysis
    pipe = construct_hyperpipe_schaefer(analysis_name, project_folder, cache_dir)
    pipe.fit(data.schaefer_network, data.fsiq, **{'covariates': covariates})
    os.remove(pipe.output_settings.pretrained_model_filename)