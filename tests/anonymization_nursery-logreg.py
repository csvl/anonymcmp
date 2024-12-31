#!/usr/bin/env python
from anonymcmp_utils import nursery_utils, results_utils
from anonymcmp_utils.testers import AnonymLRTester
import numpy as np

(x_train, y_train), (x_test, y_test) = nursery_utils.get_dataset()

preprocessor = nursery_utils.get_dataset_preprocessor() #False)

x_train_encoded = preprocessor.fit_transform(x_train)
x_test_encoded = preprocessor.transform(x_test)

k_trials = (10, 20, 50, 100, 200, 500, 1000)
epsilons = [0.2, 0.3, 0.4, 0.6, 1.0, 2.0, 5.0, 10.0]

QI = ["finance", "children", "health"]

tester = AnonymLRTester(attack_column='social', sensitive_column='social', max_iter=500)

fname_base = 'anonymization_nursery-logreg'

acc_vanilla, acc_proc = tester.perform_test(x_train, x_train_encoded, y_train, x_test_encoded, y_test, preprocessor,
                                            QI, k_trials, 'results/data/'+fname_base+'.npy', epsilons, multitest_vanilla=True,
                                            model_path='results/models/'+fname_base)

plot_path = 'results/plots/'
imfname = fname_base + '.png'
yminmax_array = np.array([[[0.6, 0.9], [0, 1.0]],
                         [[0.5, 1.0], [0, 1.0]],
                         [[0.2, 0.9], [0, 1.0]],
                         [[0.2, 0.8], [0, 1.0]],
                         [[0.5, 1.0], [0, 1.0]]])

results_utils.save_results([plot_path+'inference/'+imfname,
                            plot_path+'member_attack/'+imfname,
                            plot_path+'attrb_black_attack/'+imfname,
                            plot_path+'attrb_whiteLS_attack/'+imfname,
                            plot_path+'attrb_white_attack/'+imfname],
                           k_trials, acc_proc, acc_vanilla, 'results/data/'+fname_base+'.npy', yminmax_array, epsilons)
