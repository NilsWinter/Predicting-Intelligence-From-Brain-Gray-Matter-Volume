"""
===========================================================
Project: Predicting Intelligence from Brain Gray Matter
===========================================================
Description
-----------
This scripts implements the global (whole brain) analysis

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
from analyses.analysis_base import construct_hyperpipe
from data.data import IQData
import numpy as np
import os

analysis_name = 'whole_brain'
project_folder = '.'

# get data
data = IQData()
covariates = np.asarray([data.age, data.gender, data.handedness]).T
data.load_whole_brain(use_cached=True)

# run analysis
pipe = construct_hyperpipe(analysis_name, project_folder)
pipe.fit(data.whole_brain, data.fsiq, **{'covariates': covariates})
os.remove(pipe.output_settings.pretrained_model_filename)