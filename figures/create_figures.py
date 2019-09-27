"""
===========================================================
Project: Predicting Intelligence from Brain Gray Matter
===========================================================
Description
-----------
Process and save results. Create figures presented in the paper.

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

import numpy as np
import pandas as pd
from glob import glob
import matplotlib.pylab as plt
import seaborn as sns
from photonai.validation.ResultsTreeHandler import ResultsTreeHandler

from helper.perm_test_helper import calculate_p, metrics_list_to_dict, calculate_differences, calculate_metric_r, \
    exclude_nan

from helper.figure_collection import group_boxplot, hist_permutation_test


class Results:
    network = None
    analysis_type = None
    best_config = None
    y_true = None
    y_pred = None
    fold_indices = None
    mae = None
    mse = None
    perms = None


#----------------------
# Get and save results
#----------------------
networks = ["whole_brain", "visual", "somatomotor", "dorsal_attention", "ventral_attention", "limbic",
            "fronto_parietal", "default_mode", "subcortical", "cerebellum"]

results_folder = {'TivRescaling': '/spm-data/Scratch/spielwiese_nils_winter/iq_frankfurt/perm_test_050219',
                  'NoTivRescaling': '/spm-data/Scratch/spielwiese_nils_winter/iq_frankfurt/noTivRescaling'}

metrics = ['mean_squared_error', 'mean_absolute_error']


results = {}
for analysis, analysis_folder in results_folder.items():
    results[analysis] = {}
    df_predictions = pd.DataFrame()
    df_metrics = pd.DataFrame()
    for network in networks:
        # True run
        res = Results()
        res.network = network
        res.analysis_type = analysis

        # load results with ResultsTreeHandler
        handler = ResultsTreeHandler()
        handler.load_from_file(glob(analysis_folder + '/true_runs/' + network + '*/photon_result_file.p')[0])

        # get best config
        res.best_config = handler.results.best_config.human_readable_config

        # get predictions
        predictions = handler.get_val_preds(sort_CV=False)
        res.y_true = predictions['y_true']
        res.y_pred = predictions['y_pred']
        res.fold_indices = predictions['fold_indices']

        # get performances
        performance = handler.get_performance_outer_folds()
        res.mse = performance['mean_squared_error']
        res.rmse = np.sqrt(res.mse)
        res.mae = performance['mean_absolute_error']
        res.r = calculate_metric_r(res.y_true, res.y_pred, res.fold_indices)

        # get permutation performances
        res.perms = metrics_list_to_dict(np.load(analysis_folder + '/' + network + '_perm_results.npy'))

        results[analysis][network] = res

        # save results to dataframe and export to csv later
        df_predictions['y_pred_' + network] = res.y_pred
        df_metrics['MSE_' + network] = res.mse
        df_metrics['RMSE_' + network] = res.rmse
        df_metrics['MAE_' + network] = res.mae
        df_metrics['R_' + network] = res.r
    df_predictions['y_true'] = res.y_true
    df_predictions['fold'] = res.fold_indices
    df_predictions.to_csv('IQ_{}_predictions.csv'.format(analysis))
    df_metrics.to_csv('IQ_{}_metrics.csv'.format(analysis))


#--------------------------------------
# Calculate differences and p values
#--------------------------------------
results['differences'] = {}
for network in networks:
    perm_diff = calculate_differences(results['TivRescaling'][network].perms,
                                      results['NoTivRescaling'][network].perms, absolute=True)
    true_diff = calculate_differences({'mean_squared_error': np.mean(results['TivRescaling'][network].mse),
                                       'mean_absolute_error': np.mean(results['TivRescaling'][network].mae)},
                                      {'mean_squared_error': np.mean(results['NoTivRescaling'][network].mse),
                                       'mean_absolute_error': np.mean(results['NoTivRescaling'][network].mae)})

    p = calculate_p(true_diff, perm_diff, greater_is_better=True)
    results['differences'][network] = {'perm': perm_diff, 'true': true_diff, 'p': p}


#--------------------------------------
# Figures
#--------------------------------------

# Scatter Whole Brain
for analysis in results_folder.keys():
    sns.set()
    sns.set_style("white")
    sns.set_style("ticks")
    ax = sns.jointplot("Observed IQ", "Predicted IQ",
                       data=pd.DataFrame({'Observed IQ': results[analysis]['whole_brain'].y_true,
                                          'Predicted IQ': results[analysis]['whole_brain'].y_pred}),
                       kind="reg", stat_func=None, color='k', ylim=(40, 140), xlim=(40,140),
                       marginal_kws=dict({'hist_kws': {'edgecolor': "w", 'linewidth': 1}}))
    ax.ax_marg_x.set_title('global')

    plt.savefig('global_scatter_{}.png'.format(analysis), dpi=300)


# Scatter networks
for analysis in results_folder.keys():
    # Scatter networks
    names = ['visual network', 'somatomotor network', 'dorsal attention network', 'ventral attention network',
             'limbic network', 'fronto-parietal network', 'default-mode network', 'subcortical network', 'cerebellum']
    for i, network in enumerate(networks[1:]):
        ax = sns.jointplot("Observed IQ", "Predicted IQ",
                      data=pd.DataFrame({'Observed IQ': results[analysis][network].y_true,
                                         'Predicted IQ': results[analysis][network].y_pred}),
                      kind="reg", stat_func=None, color='k',  ylim=(40, 140), xlim=(40,140),
                      marginal_kws=dict({'hist_kws': {'edgecolor': "w", 'linewidth': 1}}))
        ax.ax_marg_x.set_title(names[i].upper())
        plt.savefig('local_scatter_{}_{}.png'.format(analysis, network), dpi=300)


# Whole Brain Box Plot and Networks Box Plot
colors = ['#5902FC', '#0270FC', '#02FCF8', '#28DC00', '#93FF09', '#FFB201', '#FF0101', '#F5C89E', '#FC69EC']
network_names = [name.replace('_', ' ').upper() for name in networks]

for analysis in results_folder.keys():
    sns.set()
    sns.set_style("white")
    sns.set_style("ticks")

    data = {'mean squared error': pd.DataFrame(),
            'mean absolute error': pd.DataFrame(),
            'root mean squared error': pd.DataFrame()}

    for network in networks:
        network_styled = network.replace('_', ' ').upper()
        data['mean squared error'][network_styled] = results[analysis][network].mse
        data['mean absolute error'][network_styled] = results[analysis][network].mae
        data['root mean squared error'][network_styled] = results[analysis][network].rmse

    for metric, df in data.items():
        # whole brain
        ax = group_boxplot(pd.DataFrame(df['WHOLE BRAIN']), ['WHOLE BRAIN'], colors=None, ylabel=metric.upper(),
                           figsize=(4, 6))
        plt.savefig('global_performance_{}_{}'.format(analysis, metric))

        # local
        group_boxplot(df[network_names[1:]], network_names[1:], colors=colors, ylabel=metric.upper())
        plt.savefig('local_performance_{}_{}'.format(analysis, metric))

        # both in one figure
        fig = plt.figure(figsize=(10, 6))
        ax1 = plt.subplot2grid((1, 5), (0, 0))
        ax2 = plt.subplot2grid((1, 5), (0, 1), colspan=4)

        group_boxplot(pd.DataFrame(df['WHOLE BRAIN']), ['WHOLE BRAIN'], colors=None, ylabel=metric.upper(), ax=ax1)
        ax1.set_xticklabels(['WHOLE BRAIN'], rotation=45, ha='right')
        ax1.set_ylabel(metric.upper())
        group_boxplot(df[network_names[1:]], network_names[1:], colors=colors, ylabel=metric.upper(), ax=ax2)
        plt.tight_layout()
        plt.savefig('global_and_local_performance_{}_{}'.format(analysis, metric))

# histograms permutation test
for metric in ['mean_squared_error', 'mean_absolute_error']:
    # differences
    sns.set()
    sns.set_style("white")
    fig, axs = plt.subplots(2, 5, figsize=(15, 7))
    axs = axs.ravel()
    for i, network in enumerate(networks):
        hist_permutation_test(results['differences'][network]['true'][metric],
                              results['differences'][network]['perm'][metric],
                              results['differences'][network]['p'][metric],
                              metric.replace('_', ' '), network.upper().replace('_', ' '), axs[i])
        axs[i].legend(loc=1, fontsize='xx-small')
    plt.tight_layout()
    plt.savefig('Permutation_Test_Differences_{}.png'.format(metric), dpi=300)
    plt.show()

#    performance
    for analysis in results_folder.keys():
        sns.set()
        sns.set_style("white")
        fig, axs = plt.subplots(2, 5, figsize=(15, 7))
        axs = axs.ravel()
        for i, network in enumerate(networks):
            if metric == 'mean_squared_error':
                perf = np.mean(results[analysis][network].mse)
            else:
                perf = np.mean(results[analysis][network].mae)
            p = calculate_p({metric: perf}, results[analysis][network].perms, False)
            hist_permutation_test(perf, exclude_nan(results[analysis][network].perms[metric]),
                                  p[metric],
                                  metric.replace('_', ' '), network.upper().replace('_', ' '), axs[i])
            axs[i].legend(loc=1, fontsize='xx-small')
        plt.tight_layout()
        plt.savefig('Permutation_Test_{}_{}.png'.format(analysis, metric), dpi=300)
        plt.show()
