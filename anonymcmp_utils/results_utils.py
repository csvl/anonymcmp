import matplotlib.pyplot as plt
import numpy as np


def save_results(fname_list, k_trials, acc_proc, acc_vanilla, npfname, yminmax_list, epsilons=None):
    np.save(npfname, (acc_vanilla, acc_proc), allow_pickle=True)

    ylabels = ['Model accuracy', 'Membership attack accuracy', 'Attribute blackbox attack accuracy',
               'Attribute whitebox attack accuracy 1', 'Attribute whitebox attack accuracy 2']

    for i in range(len(acc_vanilla[0])):
        #accs_mes = {k: np.array(accs)[:, i] for k, accs in acc_proc.items()}
        accs_mes = {k: np.array(accs)[:, :, i] for k, accs in acc_proc.items()}
        plot_anoresult(fname_list[i], k_trials, ylabels[i], accs_mes, acc_vanilla[:,i],
                       yminmax_list[i][0], yminmax_list[i][1], epsilons)


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

    legend = ax.legend(lns, labs, loc=3, framealpha=1, frameon=True)
    export_legend(legend, fname.replace('.', '_legend.'))
    legend.remove()

    fig.tight_layout()
    plt.savefig(fname)


def export_legend(legend, filename, expand=[-5,-5,5,5]):
    fig = legend.figure
    fig.canvas.draw()
    bbox = legend.get_window_extent()
    bbox = bbox.from_extents(*(bbox.extents + np.array(expand)))
    bbox = bbox.transformed(fig.dpi_scale_trans.inverted())
    fig.savefig(filename, dpi="figure", bbox_inches=bbox)
