import matplotlib
import matplotlib.pylab as plt
import numpy as np
import seaborn as sns
import pandas as pd

from scipy.stats import pearsonr
from matplotlib.markers import TICKDOWN


def scatter_predictions_per_fold(y_true, y_pred, fold_idx, file=None, show=False, ylim=None, xlim=None, dpi=300,
                                 cmap='plasma', ylabel='Predicted', xlabel='Observed', xticks=None, yticks=None,
                                 ax=None, title=None):
    plt.style.use('default')
    BIGGER_SIZE = 10
    plt.rc('font', size=BIGGER_SIZE)          # controls default text sizes
    plt.rc('axes', titlesize=BIGGER_SIZE)     # fontsize of the axes title
    plt.rc('axes', labelsize=BIGGER_SIZE)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=BIGGER_SIZE)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=BIGGER_SIZE)    # fontsize of the tick labels
    plt.rc('legend', fontsize=BIGGER_SIZE)    # legend fontsize
    plt.rc('figure', titlesize=BIGGER_SIZE)
    plt.rcParams['axes.linewidth']=1

    if not ax:
        fig, ax = plt.subplots(figsize=(5, 5), dpi=dpi)
    cm = matplotlib.cm.get_cmap(cmap)
    colors = list()
    plots = list()

    # calculate correlations
    r_values = list()
    for fold in np.unique(fold_idx):
        true = y_true[fold_idx==fold]
        pred = y_pred[fold_idx==fold]
        r_values.append(pearsonr(true,pred)[0])
    sort_idx = np.argsort(r_values)
    sorted_folds = np.unique(fold_idx)[sort_idx]
    r_values_string = list()

    for cnt, i in enumerate(sorted_folds):
        rgba = cm(cnt/10)
        true = y_true[fold_idx == i]
        pred = y_pred[fold_idx == i]
        plots.append(ax.scatter(true, pred, c=rgba, alpha=0.8, s=12))
        fit = np.polyfit(true, pred, deg=1)
        if xlim:
            xi = np.arange(xlim[0], xlim[1])
        else:
            xi = true
        ax.plot(xi, fit[0] * xi + fit[1], color=rgba, linewidth=2, alpha=0.8)
        r_values_string.append("{:.2f}".format(pearsonr(true,pred)[0]))
        colors.append(rgba)
    ax.set_ylabel(ylabel)
    ax.set_xlabel(xlabel)
    if ylim:
        ax.set_ylim(ylim[0], ylim[1])
    if xlim:
        ax.set_xlim(xlim[0],xlim[1])
    ax.grid(False)
    if xticks:
        ax.set_xticks(xticks)
    if yticks:
        ax.set_yticks(yticks)
    sns.despine(offset=5, trim=True)
    legend = ax.legend(plots,
               r_values_string,
               scatterpoints=1,
               loc='center right',
               ncol=1,
               title='r per fold',
               fontsize=7,
               bbox_to_anchor=(1.4, 0.5),
               frameon=False)
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.set_aspect('equal')
    plt.tight_layout()
    if file:
        plt.savefig(file,bbox_extra_artists=(legend,), bbox_inches='tight')
    if show:
        plt.show()
    return ax


def hist_permutation_test(score, permutation_scores, pvalue, metric_name='Score', title=None, ax=None):
    sns.set()
    sns.set_style("white")
    sns.distplot(permutation_scores, label='null distribution', hist_kws=dict(edgecolor="k", linewidth=1), ax=ax)
    ylim = ax.get_ylim()
    if pvalue == 0:
        pvalue = 0.001
        ax.plot(2 * [score], ylim, '--r', linewidth=3, label='true score (p < {})'.format(pvalue))
    else:
        ax.plot(2 * [score], ylim, '--r', linewidth=3, label='true score (p = {:.3f})'.format(pvalue))
    ax.set_ylim(ylim)
    ax.legend(loc=1, fontsize='small')
    ax.set_title(title)
    ax.set_xlabel(metric_name)
    sns.despine()


def group_boxplot(data: pd.DataFrame, names, colors, ylabel=None, ylim=None, figsize=(8,6), ax=None):

    if ax is None:
        plt.figure(figsize=figsize)
        ax = sns.boxplot(data=data, orient='v', width=0.5,
                         palette=colors, order=names)
        sns.stripplot(data=data, jitter=True,
                      palette=colors, split=True, linewidth=1, edgecolor='gray')
    else:
        ax = sns.boxplot(data=data, orient='v', width=0.5,
                         palette=colors, order=names, ax=ax)
        sns.stripplot(data=data, jitter=True,
                      palette=colors, split=True, linewidth=1, edgecolor='gray', ax=ax)
    sns.despine(offset=15)
    plt.ylabel(ylabel)
    if ylim:
        plt.ylim(ylim)
    ax.set_xticklabels(names, rotation=45, ha='right')
    ax.tick_params(axis='x', which='major', pad=0)
    plt.tight_layout()
    return ax


def significance_bar(start,end,height,displaystring,linewidth = 1.2,markersize = 8,boxpad  =0.3,fontsize = 15,color = 'k'):
    # draw a line with downticks at the ends
    plt.plot([start,end],[height]*2,'-',color = color,lw=linewidth,marker = TICKDOWN,markeredgewidth=linewidth,
             markersize = markersize)
    # draw the text with a bounding box covering up the line
    plt.text(0.5*(start+end),height,displaystring,ha = 'center',va='center',
             bbox=dict(facecolor='1.', edgecolor='none', boxstyle='Square,pad='+str(boxpad)),
             size = fontsize)

