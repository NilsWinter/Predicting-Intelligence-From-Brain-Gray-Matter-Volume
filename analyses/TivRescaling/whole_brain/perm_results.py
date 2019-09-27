"""
===========================================================
Project: IQ-Project
===========================================================
Description
-----------
Calculate p-value for whole brain analysis


Version
-------
Created:        05-02-2019
Last updated:   05-02-2019


Author
------
Nils R. Winter
nils.r.winter@gmail.com
Translationale Psychiatrie
Universitaetsklinikum Muenster
"""

from analyses.perm_test_helper import load_perm_results_from_folder, calculate_p


results_folder = '/scratch/tmp/wintern/iq_prediction/results/perm/whole_brain/'
pipe_name = 'whole_brain'

perm_results, best_configs = load_perm_results_from_folder(pipe_name, results_folder)

