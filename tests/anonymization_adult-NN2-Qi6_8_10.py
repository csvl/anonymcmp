#!/usr/bin/env python
from anonymcmp_utils import adult_utils, results_utils
from anonymcmp_utils.testers import AnonymGBClass2Tester
import numpy as np

#(x_train, y_train), (x_test, y_test) = adult_utils.get_dataset_bin_relationship()

from apt.utils.dataset_utils import get_adult_dataset_pd
(x_train, y_train), (x_test, y_test) = get_adult_dataset_pd()

preprocessor = adult_utils.get_dataset_preprocessor(x_train)

x_train_encoded = preprocessor.fit_transform(x_train)
x_test_encoded = preprocessor.transform(x_test)

#k_trials = (50, 100, 200, 400, 800, 1000)
k_trials = (10, 20, 50, 100, 200, 500, 1000)

QIs = [['age', 'education-num', 'marital-status', 'race', 'sex', 'native-country'],
       ['age', 'workclass', 'education-num', 'marital-status', 'occupation', 'race', 'sex', 'native-country'],
       ['age', 'workclass', 'education-num', 'marital-status', 'occupation', 'relationship', 'race', 'sex',
        'hours-per-week', 'native-country']]

tester = AnonymGBClass2Tester(attack_column='relationship', sensitive_column='relationship',
                              input_veclen=x_train_encoded.shape[1], sample_len=len(x_train))

acc_vanilla, acc_proc_list = tester.perform_anonymtest_QIs(x_train, x_train_encoded, y_train, x_test_encoded, y_test,
                                                           preprocessor, QIs, k_trials,
                                                           'results/models/anonymization_adult-NN2/vanilla_0',
                                                           'results/data/anonymization_adult-NN2-Qi',
                                                           model_root_path='results/models/anonymization_adult-NN2-Qi')

plot_path = 'results/plots/'
minmax_array = np.array([[[0.6, 0.9], [0.55, 1.0]],
                         [[0.5, 1.0], [0.45, 1.0]],
                         [[0.2, 0.9], [0.45, 1.0]],
                         [[0.2, 0.8], [0.45, 1.0]],
                         [[0.7, 1.0], [0, 1.0]]])


def save_results(QI, acc_proc):
    fname_base = 'anonymization_adult-NN2-Qi' + str(len(QI))
    imfname = fname_base + '.png'

    results_utils.save_results([plot_path+'inference/'+imfname,
                                plot_path+'member_attack/'+imfname,
                                plot_path+'attrb_black_attack/'+imfname],
                               k_trials, acc_proc, acc_vanilla, 'results/data/'+fname_base+'.npy', minmax_array)

for QI, acc_proc in zip(QIs, acc_proc_list):
    save_results(QI, acc_proc)
