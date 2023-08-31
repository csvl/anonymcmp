#!/usr/bin/env python
from anonymcmp_utils import nursery_utils, results_utils
from anonymcmp_utils.testers import AnonymRFTester

(x_train, y_train), (x_test, y_test) = nursery_utils.get_dataset()

preprocessor = nursery_utils.get_dataset_preprocessor(False)

x_train_encoded = preprocessor.fit_transform(x_train)
x_test_encoded = preprocessor.transform(x_test)

k_trials = (50, 100, 200, 400, 800, 1000)
epsilons = [0.01, 0.1, 0.5, 1.0, 5.0, 10.0, 25.0, 75.0, 100.0, 200.0]

QI = ["finance", "social", "health"]

tester = AnonymRFTester(attack_column='social', sensitive_column='health')

acc_vanilla, acc_proc = tester.perform_test(x_train, x_train_encoded, y_train, x_test_encoded, y_test, preprocessor,
                                            QI, k_trials, epsilons)

plot_path = 'results/plots/'
fname_base = 'anonymization_nursery-randforest'
imfname = fname_base + '.png'

yminmax_list = [[0.35, 1.0], [0.35, 1.0], [0.35, 1.0], [0.35, 1.0], [0.35, 1.0]]

results_utils.save_results([plot_path+'inference/'+imfname,
                            plot_path+'member_attack/'+imfname,
                            plot_path+'attrb_black_attack/'+imfname,
                            plot_path+'attrb_whiteLS_attack/'+imfname,
                            plot_path+'attrb_white_attack/'+imfname],
                           k_trials, acc_proc, acc_vanilla, 'results/data/'+fname_base+'.npy', yminmax_list, epsilons)
