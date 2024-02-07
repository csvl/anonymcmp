import numpy as np
from anonymcmp_utils import results_utils

acc_vanilla, acc_proc = np.load('data/anonymization_adult-NN1.npy', allow_pickle=True)

acc_proc.pop('Differential privacy')

k_trials = (50, 100, 200, 400, 800, 1000)

fname_base = 'anonymization_adult-NN1_withoutDP'
imfname = fname_base + '.png'
yminmax_list = [[0.6, 0.9], [0.55, 1.0], [0.55, 1.0], [0.55, 1.0], [0.55, 1.0]]

fname_list = ['plots/inference/'+imfname,
              'plots/member_attack/'+imfname,
              'plots/attrb_black_attack/'+imfname]

ylabels = ['Model accuracy', 'Membership attack accuracy', 'Attribute blackbox attack accuracy',
           'Attribute whitebox attack accuracy 1', 'Attribute whitebox attack accuracy 2']

for i in range(len(acc_vanilla[0])):
    accs_mes = {k: np.array(accs)[:, :, i] for k, accs in acc_proc.items()}
    results_utils.plot_anoresult(fname_list[i], k_trials, ylabels[i], accs_mes, acc_vanilla[:,i],
                   yminmax_list[i][0], yminmax_list[i][1])
