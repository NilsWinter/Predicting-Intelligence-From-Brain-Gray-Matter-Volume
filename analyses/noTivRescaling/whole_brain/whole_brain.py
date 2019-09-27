"""
===========================================================
Project: IQ-Prediction Frankfurt
===========================================================
Description
-----------
This scripts implements whole brain analysis with NoTivRescaling
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
from analyses.analysis_base import construct_hyperpipe
from data.data import IQData
import numpy as np
import os


analysis_name = 'whole_brain_no_tiv'
data_folder = '/scratch/tmp/wintern/iq_frankfurt/'
project_folder = '/scratch/tmp/wintern/iq_frankfurt/results/noTivRescaling/whole_brain'
cache_dir = '/scratch/tmp/wintern/cache'

# get data
data = IQData(data_folder=data_folder, tiv_rescaled=False)
covariates = np.asarray([data.age, data.gender, data.handedness]).T
data.load_whole_brain(use_cached=False)

# run analysis
pipe = construct_hyperpipe(analysis_name, project_folder, cache_dir)
pipe.fit(data.whole_brain, data.fsiq, **{'covariates': covariates})
os.remove(pipe.output_settings.pretrained_model_filename)