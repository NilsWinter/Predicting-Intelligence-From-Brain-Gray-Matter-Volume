"""
===========================================================
Project: IQ-Prediction Frankfurt
===========================================================
Description
-----------
This scripts implements shen atlas analysis

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


analysis_name = 'shen'
data_folder = '/scratch/tmp/wintern/iq_frankfurt/'
project_folder = '/scratch/tmp/wintern/iq_frankfurt/results/TivRescaling/shen/'
cache_dir = '/scratch/tmp/wintern/cache'

# get data
data = IQData(data_folder=data_folder)
covariates = np.asarray([data.age, data.gender, data.handedness]).T
data.load_shen(use_cached=False)

# run analysis
pipe = construct_hyperpipe_schaefer(analysis_name, project_folder, cache_dir)
pipe.fit(data.shen, data.fsiq, **{'covariates': covariates})
os.remove(pipe.output_settings.pretrained_model_filename)