"""
===========================================================
Project: IQ-Project
===========================================================
Description
-----------
Analyze all results for IQ project. Do visualizations.

Version
-------
Created:        08-03-2019
Last updated:   08-03-2019


Author
------
Nils R. Winter
nils.r.winter@gmail.com
Translationale Psychiatrie
Universitaetsklinikum Muenster
"""

import numpy as np
import pandas as pd
from analyses.perm_test_helper import load_perm_results_from_folder, calculate_p, metrics_list_to_dict, load_results
import glob
import os
from DemoFiles.Nils.figures.figure_collection import *
import matplotlib.pylab as plt
import seaborn as sns
from photonai.validation.ResultsTreeHandler import ResultsTreeHandler


def exclude_nan(x):
    return [i for i in x if i is not None]


networks = ["whole_brain", "visual", "somatomotor", "dorsal_attention", "ventral_attention", "limbic", "fronto_parietal",
            "default_mode", "subcortical", "cerebellum"]



results_folder = {'TivRescaling': '/spm-data/Scratch/spielwiese_nils_winter/iq_frankfurt/perm_test_050219',
                  'NoTivRescaling': '/spm-data/Scratch/spielwiese_nils_winter/iq_frankfurt/noTivRescaling'}

for analysis in results_folder.keys():
    true_performance = dict()
    ps = list()
    # networks
    for metric_name in ['mean_squared_error', 'mean_absolute_error']:
        sns.set()
        sns.set_style("white")
        fig, axs = plt.subplots(2, 5, figsize=(15, 7))
        axs = axs.ravel()
        for i, network in enumerate(networks):
            perm_results = np.load(results_folder[analysis] + '/' + network + '_perm_results.npy')
            true_performance[network], best_config = load_results(glob.glob(results_folder[analysis] + '/true_runs/' + network +
                                                                                '*/photon_result_file.p')[0])

            perm_results = metrics_list_to_dict(perm_results)
            ps.append(calculate_p(true_performance[network], perm_results, False))
            hist_permutation_test(true_performance[network][metric_name], exclude_nan(perm_results[metric_name]),
                                      ps[i][metric_name],
                                      metric_name.replace('_', ' '), network.upper().replace('_', ' '), axs[i])
            axs[i].legend(loc=1, fontsize='xx-small')
        plt.tight_layout()
        plt.savefig('IQ_Prediction_{}_{}.png'.format(metric_name, analysis), dpi=300)
        plt.show()

#--------------------
# DIFFERENCES
#--------------------
def calculate_differences(tiv, notiv, sort=False):
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
                    diff[metric].append(tiv[metric][run] - notiv[metric][run])
        else:
            if tiv[metric] and notiv[metric]:
                diff[metric] = tiv[metric] - notiv[metric]
    return diff

# networks
for metric_name in ['mean_squared_error', 'mean_absolute_error']:
    sns.set()
    sns.set_style("white")
    fig, axs = plt.subplots(2, 5, figsize=(15, 7))
    axs = axs.ravel()
    for i, network in enumerate(networks):
        perm_tiv = np.load(results_folder['TivRescaling'] + '/' + network + '_perm_results.npy')
        perm_notiv = np.load(results_folder['NoTivRescaling'] + '/' + network + '_perm_results.npy')

        true_tiv, best_config_tiv = load_results(glob.glob(results_folder['TivRescaling'] + '/true_runs/' + network +
                                                                            '*/photon_result_file.p')[0])
        true_notiv, best_config_notiv = load_results(glob.glob(results_folder['NoTivRescaling'] + '/true_runs/' + network +
                                                                            '*/photon_result_file.p')[0])
        perm_tiv = metrics_list_to_dict(perm_tiv)
        perm_notiv = metrics_list_to_dict(perm_notiv)
        perm_diff = calculate_differences(perm_tiv, perm_notiv)
        true_diff = calculate_differences(true_tiv, true_notiv)
        p = calculate_p(true_diff, perm_diff, False)
        hist_permutation_test(true_diff[metric_name], perm_diff[metric_name],
                                  p[metric_name],
                                  metric_name.replace('_', ' '), network.upper().replace('_', ' '), axs[i])
        axs[i].legend(loc=1, fontsize='xx-small')
    plt.tight_layout()
    plt.savefig('IQ_Prediction_Differences_{}.png'.format(metric_name), dpi=300)
    plt.show()

# colors
"""
visual module = #5902FC
somatomotor module = #0270FC
dorsal attention module = #02FCF8
ventral attention module = #28DC00
limbic module = #A3FF01
fronto-patietal module = #FFB201
default-mode module = #FF0101
subcortical module = #FF9898
cerebellum = #FC69EC
"""

tips = sns.load_dataset("tips")


colors = ['#646464', '#5902FC', '#0270FC', '#02FCF8', '#28DC00', '#93FF09', '#FFB201', '#FF0101', '#F5C89E', '#FC69EC']
dummy_results = {'mean_squared_error': (166.93, 14.40), 'mean_absolute_error': (10.48, 0.32)}
sns.set()
sns.set_style("white")
for metric_name in ['mean_squared_error', 'mean_absolute_error']:
    results_modules = pd.DataFrame()
    true_performance_std = dict()
    for name in networks:
        tiv = '/spm-data/Scratch/spielwiese_nils_winter/iq_frankfurt/perm_test_050219'
        notiv = '/spm-data/Scratch/spielwiese_nils_winter/iq_frankfurt/noTivRescaling'
        handler = ResultsTreeHandler()
        handler.load_from_file(glob.glob(tiv + '/true_runs/' + name + '*/photon_result_file.p')[0])
        perfs_tiv = handler.get_performance_outer_folds()
        handler = ResultsTreeHandler()
        handler.load_from_file(glob.glob(notiv + '/true_runs/' + name + '*/photon_result_file.p')[0])
        perfs_notiv = handler.get_performance_outer_folds()
        perfs_tiv[metric_name].extend(perfs_notiv[metric_name])
        results_modules[metric_name + '-' + name] = perfs_tiv[metric_name]
    group = ['with TIV rescaling']*10
    group.extend(['without TIV rescaling']*10)
    results_modules['scaling'] = group
    results_modules['id'] = results_modules.index
    results_modules.to_csv('IQ_prediction_results_{}_14-03-19.csv'.format(metric_name))
    results_long = pd.wide_to_long(results_modules, stubnames=metric_name, j="networks", i="id", sep='-', suffix="\w+")
    results_long.reset_index(inplace=True)
    plt.figure(figsize=(10, 6))
    ax = sns.boxplot(x="networks", y=metric_name, data=results_long, hue='scaling')
    sns.stripplot(x="networks", y=metric_name, data=results_long, hue='scaling', jitter=True,
                  split=True, linewidth=1, edgecolor='gray')
    sns.despine(offset=15)
    #plt.ylabel(networks)
    ax.set_xticklabels(networks, rotation=45, ha='right')
    ax.tick_params(axis='x', which='major', pad=0)
    handles, labels = ax.get_legend_handles_labels()

    # When creating the legend, only use the first two elements
    # to effectively remove the last two.
    l = plt.legend(handles[0:2], labels[0:2], bbox_to_anchor=(1, 1), loc=1, borderaxespad=0.)
    plt.tight_layout()

    ax.axhline(dummy_results[metric_name][0], color='#9E9E9E', lw=1)
    ax.axhline(dummy_results[metric_name][0] + dummy_results[metric_name][1], color='#9E9E9E', lw=1, linestyle='--')
    ax.axhline(dummy_results[metric_name][0] - dummy_results[metric_name][1], color='#9E9E9E', lw=1, linestyle='--')

    plt.savefig('IQ_prediction_performance_{}.png'.format(metric_name), dpi=300)



sns.set()
sns.set_style("white")
results_modules = pd.DataFrame()



for analysis in results_folder.keys():
    # for analysis in results_folder.keys():
    sns.set()
    sns.set_style("white")
    fig, axs = plt.subplots(5, 2, figsize=(9, 15), sharex='all', sharey='all')
    axs = axs.ravel()
    for i, name in enumerate(networks):
        handler = ResultsTreeHandler()
        handler.load_from_file(glob.glob(results_folder[analysis] + '/true_runs/' + name + '*/photon_result_file.p')[0])
        preds = handler.get_val_preds(sort_CV=False)
        ax = scatter_predictions_per_fold(y_true=preds['y_true'], y_pred=preds['y_pred'],
                                     fold_idx=preds['fold_indices'],
                                     xticks=[60, 80, 100, 120, 140], yticks=[60, 80, 100, 120, 140],
                                     ylim=(50, 150), xlim=(50, 150), ylabel='Predicted FSIQ', xlabel='Observed FSIQ',
                                     ax=axs[i], title=name.upper())

    plt.suptitle(analysis.upper())
    plt.savefig('IQ_Project_scatter_predictions_{}.png'.format(analysis))






