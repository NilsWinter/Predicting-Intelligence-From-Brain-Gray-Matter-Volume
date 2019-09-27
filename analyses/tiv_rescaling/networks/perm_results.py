"""
===========================================================
Project: IQ-Project
===========================================================
Description
-----------
Calculate p-value for network analysis


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

from helper.perm_test_helper import load_perm_results_from_folder, calculate_p, metrics_list_to_dict, load_results
import numpy as np
import glob
from os import path
import os


networks = ["whole_brain", "visual", "somatomotor", "dorsal_attention", "ventral_attention", "limbic", "fronto_parietal",
            "default_mode", "subcortical", "cerebellum"]
results_folder = '/spm-data/Scratch/spielwiese_nils_winter/iq_frankfurt/perm_test_050219'

log_file = open(path.join("results_perm_test.log"), "w")

for network in networks:
    perm_results, _ = load_perm_results_from_folder(network, results_folder)
    np.save(results_folder + '/' + network + '_perm_results.npy', perm_results)
    #perm_results = np.load(results_folder + '/' + network + '_perm_results.npy')

    for true_run_folder in glob.glob(results_folder + '/true_runs/' + network + '*'):
        true_performance, best_config = load_results(os.path.join(true_run_folder, 'photon_result_file.p'))
    p = calculate_p(true_performance, metrics_list_to_dict(perm_results), False)
    log_file.write("ROI {}\n".format(network))
    log_file.write("MSE = {:.2f}  p = {:.4f}\n".format(true_performance['mean_squared_error'], p['mean_squared_error']))
    log_file.write("MAE = {:.2f}  p = {:.4f}\n\n".format(true_performance['mean_absolute_error'], p['mean_absolute_error']))
log_file.close()

