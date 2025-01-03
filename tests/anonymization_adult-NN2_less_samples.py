#!/usr/bin/env python
from anonymcmp_utils import adult_utils, results_utils
from anonymcmp_utils.testers import AnonymGBClass2Tester
import numpy as np

#(x_train, y_train), (x_test, y_test) = adult_utils.get_dataset_bin_relationship()
from apt.utils.dataset_utils import get_adult_dataset_pd
(x_train, y_train), (x_test, y_test) = get_adult_dataset_pd()

def run_with_less_samples(dev_factor, fname_base, orig_x_train, orig_y_train):
    x_train = orig_x_train[:int(len(orig_x_train)/dev_factor)]
    y_train = orig_y_train[:int(len(orig_y_train)/dev_factor)]

    preprocessor = adult_utils.get_dataset_preprocessor(x_train)

    x_train_encoded = preprocessor.fit_transform(x_train)
    x_test_encoded = preprocessor.transform(x_test)

    k_trials = (10, 20, 50, 100, 200, 500, 1000)
    epsilons = [0.2, 0.3, 0.4, 0.6, 1.0, 2.0, 5.0, 10.0]
    QI = ['age', 'education-num', 'race', 'native-country']

    tester = AnonymGBClass2Tester(attack_column='relationship', sensitive_column='relationship',
                                  input_veclen=x_train_encoded.shape[1], sample_len=len(x_train))

    acc_vanilla, acc_proc = tester.perform_test(x_train, x_train_encoded, y_train, x_test_encoded, y_test, preprocessor,
                                                QI, k_trials, 'results/data/'+fname_base+'.npy', epsilons,
                                                multitest_vanilla=True, model_path='results/models/' + fname_base)

    plot_path = 'results/plots/'
    imfname = fname_base + '.png'

    yminmax_array = np.array([[[0.6, 0.9], [0, 1.0]],
                              [[0.5, 1.0], [0, 1.0]],
                              [[0.2, 0.9], [0, 1.0]],
                              [[0.2, 0.8], [0, 1.0]],
                              [[0.7, 1.0], [0, 1.0]]])

    results_utils.save_results([plot_path+'inference/'+imfname,
                                plot_path+'member_attack/'+imfname,
                                plot_path+'attrb_black_attack/'+imfname],
                               k_trials, acc_proc, acc_vanilla, 'results/data/'+fname_base+'.npy', yminmax_array, epsilons)

list_dev_factor = [2, 4, 8, 16]
list_fname_base = ['anonymization_adult-NN2_half_samples',
                   'anonymization_adult-NN2_quart_samples',
                   'anonymization_adult-NN2_8th_samples',
                   'anonymization_adult-NN2_16th_samples']


for dev_factor, fname_base in zip(list_dev_factor, list_fname_base):
    run_with_less_samples(dev_factor, fname_base, x_train, y_train)
