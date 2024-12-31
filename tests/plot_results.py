#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt

k_trials = (10, 20, 50, 100, 200, 500, 1000)
#k_trials = (50, 100, 200, 400, 800, 1000)
#epsilons = [0.2, 0.25, 0.3, 0.35, 0.4, 0.5, 1.0, 2.5, 5.0, 7.5, 10.0]
epsilons = [0.2, 0.3, 0.4, 0.6, 1.0, 2.0, 5.0, 10.0]

fname_base = 'anonymization_adult-dectree'
#fname_base = 'anonymization_adult-dectree-Qi8'
#fname_base = 'anonymization_adult-NN1'
#fname_base = 'anonymization_adult-NN2'
#fname_base = 'anonymization_adult-logreg'
path_base = '' #'/tf/dev/'
plot_path = path_base + 'tests/results/plots/'
imfname = fname_base + '.png'
yminmax_array = np.array([[[0.5, 1.0], [0.45, 1.0]],
                         [[0.5, 1.0], [0.45, 1.0]],
                         [[0.2, 0.9], [0.45, 1.0]],
                         [[0.2, 0.8], [0.45, 1.0]],
                         [[0.5, 1.0], [0, 1.0]]])

def plot_results(fname_list, ylabels, yminmax_array, acc_vanilla, acc_proc, mes_id, k_trials, epsilons, prefix=None):
    for i in range(len(acc_vanilla[0])):
        accs_mes = {k: np.array(accs)[:, :, i, mes_id] for k, accs in acc_proc.items()}
        fname = fname_list[i] if prefix is None else fname_list[i].replace('.png', '_' + prefix + '.png')
        plot_anoresult(fname, k_trials, ylabels[i],
                       accs_mes, acc_vanilla[:, i, mes_id],
                       yminmax_array[mes_id, i, 0], yminmax_array[mes_id, i, 1], epsilons)


def plot_all_results(fname_list, k_trials, acc_proc, acc_vanilla, yminmax_array, epsilons=None):
    ylabels = ['Model accuracy', 'Membership attack accuracy', 'Attribute blackbox attack accuracy',
               'Attribute whitebox attack accuracy 1', 'Attribute whitebox attack accuracy 2']

    plot_results(fname_list, ylabels, yminmax_array, acc_vanilla, acc_proc, 0, k_trials, epsilons)

    plot_results(fname_list,
                 ['Model precision', 'Membership attack precision'],
                 yminmax_array, acc_vanilla, acc_proc, 1, k_trials, epsilons, prefix='prec')


    plot_results(fname_list,
                 ['Model recall', 'Membership attack recall'],
                 yminmax_array, acc_vanilla, acc_proc, 2, k_trials, epsilons, prefix='recall')

    plot_results(fname_list,
                 ['Model F1-score', 'Membership attack F1-score'],
                 yminmax_array, acc_vanilla, acc_proc, 3, k_trials, epsilons, prefix='f1')

    plot_results(fname_list,
                 ['Model roc auc', 'Membership attack roc auc'],
                 yminmax_array, acc_vanilla, acc_proc, 4, k_trials, epsilons, prefix='auc')


def plot_mean_std(ax, x, acc, m, k, color=None):
    accmean = acc.mean(axis=1)
    if color is None:
        ln = ax.plot(x, accmean, m, fillstyle='none', label=k)[0]
        ax.fill_between(x, accmean + acc.std(axis=1), accmean - acc.std(axis=1), color=ln.get_color(), alpha=0.5)
    else:
        ln = ax.plot(x, accmean, m, fillstyle='none', label=k, color=color)[0]
        ax.fill_between(x, accmean + acc.std(axis=1), accmean - acc.std(axis=1), color=color, alpha=0.5)

    return ln

def plot_anoresult(fname, k_trials, ylabel, acc_proc, acc_vanilla, ylim_min, ylim_max, epsilons=None):
    fig, ax = plt.subplots()

    markers = ['-s', '-^', '-o', '-<']

    lns_ano = [plot_mean_std(ax, k_trials, acc, m, k) for (k, acc), m
               in zip(acc_proc.items(), markers) if k != 'Differential privacy']

    ax.set_xscale('log')
    lns_vanilla = plot_mean_std(ax, k_trials, np.dot(np.ones_like(k_trials)[:,np.newaxis], acc_vanilla[np.newaxis,:]),
                               '--', "Base line", color='black')

    ax.set_xlabel("k")
    ax.set_ylim(ylim_min, ylim_max)

    lns = lns_ano + [lns_vanilla]

    if 'Differential privacy' in acc_proc.keys():
        ax2 = ax.twiny()
        ax2.set_xscale('log')
        ln_dp = plot_mean_std(ax2, epsilons, acc_proc['Differential privacy'], '-v', "Differntial privacy",
                              color='tab:purple')
        ax2.set_xlabel("epsilons")
        ax2.xaxis.set_label_position('top')
        ax2.xaxis.tick_top()
        ax2.set_ylim(ylim_min, ylim_max)

        lns = lns + [ln_dp]

    ax.set_ylabel(ylabel)

    labs = [l.get_label() for l in lns]

    #legend = ax.legend(lns, labs, loc=3, framealpha=1, frameon=True)
    #export_legend(legend, fname.replace('.', '_legend.'))
    #legend.remove()

    fig.tight_layout()
    plt.savefig(fname)


def export_legend(legend, filename, expand=[-5,-5,5,5]):
    fig = legend.figure
    fig.canvas.draw()
    bbox = legend.get_window_extent()
    bbox = bbox.from_extents(*(bbox.extents + np.array(expand)))
    bbox = bbox.transformed(fig.dpi_scale_trans.inverted())
    fig.savefig(filename, dpi="figure", bbox_inches=bbox)


acc_vanilla, acc_proc = np.load(path_base + 'tests/results/data/'+fname_base+'.npy', allow_pickle=True)

#acc_vanilla = np.array([[r[i] if i==0 else list(r[i])+[0.0] for i in range(len(r))] for r in acc_vanilla])

#acc_proc = {k: np.array([[[acc_k_i[0], list(acc_k_i[1])+[0.0]] for acc_k_i in acc_k] for acc_k in accs])
#            for k, accs in acc_proc.items()}

plot_all_results([plot_path+'inference/'+imfname,
              plot_path+'member_attack/'+imfname,
              plot_path+'attrb_black_attack/'+imfname,
              plot_path+'attrb_whiteLS_attack/'+imfname,
              plot_path+'attrb_white_attack/'+imfname],
              k_trials, acc_proc, acc_vanilla, yminmax_array, epsilons)


