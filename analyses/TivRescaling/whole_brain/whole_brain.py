"""
===========================================================
Project: IQ-Prediction Frankfurt
===========================================================
Description
-----------
This scripts implements all three analyses: whole brain, single networks, stacked networks

Version
-------
Created:        28-01-2019
Last updated:   28-01-2019

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


analysis_name = 'whole_brain'
data_folder = '/scratch/tmp/wintern/iq_frankfurt/'
project_folder = '/scratch/tmp/wintern/iq_frankfurt/results'
cache_dir = '/scratch/tmp/wintern/cache'

# get data
data = IQData(data_folder=data_folder)
covariates = np.asarray([data.age, data.gender, data.handedness]).T
data.load_whole_brain(use_cached=True)

# run analysis
pipe = construct_hyperpipe(analysis_name, project_folder, cache_dir)
pipe.fit(data.whole_brain, data.fsiq, **{'covariates': covariates})
os.remove(pipe.output_settings.pretrained_model_filename)