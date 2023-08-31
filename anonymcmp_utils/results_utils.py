import matplotlib.pyplot as plt
import numpy as np


def save_results(fname_list, k_trials, acc_proc, acc_vanilla, npfname, yminmax_list, epsilons=None):
    np.save(npfname, (acc_vanilla, acc_proc), allow_pickle=True)

    ylabels = ['Model accuracy', 'Membership attack accuracy', 'Attribute blackbox attack accuracy',
               'Attribute whitebox attack accuracy 1', 'Attribute whitebox attack accuracy 2']

    legend_list = [True, False, False, False, False]

    for i in range(len(acc_vanilla)):
        accs_mes = {k: np.array(accs)[:, i] for k, accs in acc_proc.items()}
        plot_anoresult(fname_list[i], k_trials, ylabels[i], accs_mes, acc_vanilla[i],
                       yminmax_list[i][0], yminmax_list[i][1], legend_list[i], epsilons)


def plot_anoresult(fname, k_trials, ylabel, acc_proc, acc_vanilla, ylim_min, ylim_max, legend, epsilons):
    fig, ax = plt.subplots()

    markers = ['-s', '-^', '-o', '-<']

    lns_ano = [ax.plot(k_trials, acc, m, fillstyle='none', label=k)[0] for (k, acc), m
               in zip(acc_proc.items(), markers) if k != 'Differential privacy']

    lns_vanilla = ax.plot(k_trials, np.ones_like(k_trials) * acc_vanilla, '--', label="Base line", color='black')[0]

    ax.set_xlabel("k")
    ax.set_ylim(ylim_min, ylim_max)

    lns = lns_ano + [lns_vanilla]

    if 'Differential privacy' in acc_proc.keys():
        ax2 = ax.twiny()
        ln_dp = ax2.plot(epsilons, acc_proc['Differential privacy'], '-v', fillstyle='none',
                         label="Differntial privacy", color='tab:purple')[0]
        ax2.set_xlabel("epsilons")
        ax2.xaxis.set_label_position('top')
        ax2.xaxis.tick_top()
        ax2.set_ylim(ylim_min, ylim_max)

        lns = lns + [ln_dp]

    ax.set_ylabel(ylabel)

    labs = [l.get_label() for l in lns]
    if legend:
        ax.legend(lns, labs)

    fig.tight_layout()
    plt.savefig(fname)
