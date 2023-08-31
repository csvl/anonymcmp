#!/usr/bin/env python
from anonymcmp_utils import adult_utils, results_utils
from anonymcmp_utils.testers import AnonymDTTester

(x_train, y_train), (x_test, y_test) = adult_utils.get_dataset_bin_relationship()

preprocessor = adult_utils.get_dataset_preprocessor(x_train, False)

x_train_encoded = preprocessor.fit_transform(x_train)
x_test_encoded = preprocessor.transform(x_test)

k_trials = (50, 100, 200, 400, 800, 1000)
epsilons = [0.01, 0.1, 0.5, 1.0, 5.0, 10.0, 25.0, 75.0, 100.0, 200.0]

QI = ['age', 'education-num', 'race', 'relationship']

tester = AnonymDTTester(attack_column='relationship', sensitive_column='relationship')

acc_vanilla, acc_proc = tester.perform_test(x_train, x_train_encoded, y_train, x_test_encoded, y_test, preprocessor,
                                            QI, k_trials, epsilons)

plot_path = 'results/plots/'
fname_base = 'anonymization_adult-dectree'
imfname = fname_base + '.png'
yminmax_list = [[0.6, 0.85], [0.55, 1.0], [0.55, 1.0], [0.55, 1.0], [0.55, 1.0]]

results_utils.save_results([plot_path+'inference/'+imfname,
                            plot_path+'member_attack/'+imfname,
                            plot_path+'attrb_black_attack/'+imfname,
                            plot_path+'attrb_whiteLS_attack/'+imfname,
                            plot_path+'attrb_white_attack/'+imfname],
                           k_trials, acc_proc, acc_vanilla, 'results/data/'+fname_base+'.npy', yminmax_list, epsilons)
