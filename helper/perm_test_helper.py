"""
===========================================================
Project: Predicting Intelligence from Brain Gray Matter
===========================================================
Description
-----------
Helper functions for collecting the results of the distributed permutation test

PHOTON Branch
-------------
project/IQ_prediction_280119

Version
-------
Created:        28-01-2019
Last updated:   27-09-2019

Author
------
Nils R. Winter
nils.r.winter@gmail.com
Translationale Psychiatrie
Universitaetsklinikum Muenster
"""

import glob
import os

import numpy as np
from scipy.stats import pearsonr

from photonai.validation.ResultsDatabase import MDBHelper


def load_perm_results_from_folder(pipe_name, results_folder):
    name = pipe_name
    perm_results = list()
    best_configs = list()
    valid = list()
    for i in range(1000):
        try:
            folder = glob.glob(results_folder + '/' + name + '_perm_{}_*'.format(i))
            if len(folder) > 1:
                print('Found multiple result folders. Skipping load process.')
                continue
            res, bc = load_results(os.path.join(folder[0], 'photon_result_file.p'))
            perm_results.append(res)
            best_configs.append(bc)
            print('Load perm result {}'.format(folder[0]))
            valid.append(1)
        except:
            print('Couldnt load perm result {}'.format(results_folder + '/' + name + '_perm_{}_*'.format(i)))
            perm_results.append(None)
            best_configs.append(None)
    print('Done loading {} perm results. No of valid perms: {}'.format(len(perm_results), len(valid)))
    return perm_results, best_configs


def load_results(local_file, extract_std=False):
    result_tree = MDBHelper().load_results(local_file)
    # collect test set predictions
    perm_performances = dict()
    try:
        best_config = result_tree.best_config.human_readable_config
    except:
        best_config = None
    if extract_std:
        for metric in result_tree.metrics_test:
            if metric.operation == "FoldOperations.STD":
                perm_performances[metric.metric_name] = metric.value
    else:
        for metric in result_tree.metrics_test:
            if metric.operation == "FoldOperations.MEAN":
                perm_performances[metric.metric_name] = metric.value
    return perm_performances, best_config


def calculate_p(true_performance, perm_performances, greater_is_better):
    p = dict()
    # Get all specified metrics
    metrics = dict()
    for metric in true_performance.keys():
        metrics[metric] = {'name': metric, 'greater_is_better': greater_is_better}
    for _, metric in metrics.items():
        n_perms = len(perm_performances[metric['name']])
        print('Number of permutations: ', n_perms)
        comparisons = list()
        for run in range(n_perms):
            if metric['greater_is_better']:
                if perm_performances[metric['name']][run]:
                    comparisons.append(true_performance[metric['name']] < np.asarray(perm_performances[metric['name']][run]))
                else:
                    n_perms -= 1
            else:
                if perm_performances[metric['name']][run]:
                    comparisons.append(true_performance[metric['name']] > np.asarray(perm_performances[metric['name']][run]))
                else:
                    n_perms -= 1

        p[metric['name']] = np.sum(comparisons) / (n_perms + 1)
    return p


def metrics_list_to_dict(perm_results):
    metric_names = None
    i = 0
    while not metric_names:
        try:
            metric_names = perm_results[i].keys()
            i += 1
        except:
            pass
    perf = dict()
    for metric in metric_names:
        perf[metric] = list()
    for perm in perm_results:
        for metric in metric_names:
            if not perm:
                perf[metric].append(None)
            else:
                perf[metric].append(perm[metric])
    return perf


def calculate_differences(tiv, notiv, sort=False, absolute=True):
    diff = dict()
    for metric in tiv.keys():
        diff[metric] = list()
        if sort:
            tiv[metric] = np.asarray(tiv[metric])
            notiv[metric] = np.asarray(notiv[metric])
            ind = tiv[metric] == None
            tiv[metric][ind] = 0
            ind = notiv[metric] == None
            notiv[metric][ind] = 0

            tiv[metric] = np.sort(tiv[metric])
            notiv[metric] = np.sort(notiv[metric])

        if isinstance(tiv[metric], list) or isinstance(tiv[metric], np.ndarray):
            for run in range(len(tiv[metric])):
                if tiv[metric][run] and notiv[metric][run]:
                    if absolute:
                        diff[metric].append(abs(tiv[metric][run] - notiv[metric][run]))
                    else:
                        diff[metric].append(tiv[metric][run] - notiv[metric][run])
        else:
            if tiv[metric] and notiv[metric]:
                if absolute:
                    diff[metric] = abs(tiv[metric] - notiv[metric])
                else:
                    diff[metric] = tiv[metric] - notiv[metric]
    return diff


def calculate_metric_r(y_true, y_pred, fold_indices):
    r_values = list()
    for fold in np.unique(fold_indices):
        true = y_true[fold_indices==fold]
        pred = y_pred[fold_indices==fold]
        r_values.append(pearsonr(true,pred)[0])
    return r_values


def exclude_nan(x):
    return [i for i in x if i is not None]