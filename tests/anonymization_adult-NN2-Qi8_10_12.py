#!/usr/bin/env python
from anonymcmp_utils import adult_utils, results_utils
from anonymcmp_utils.testers import AnonymGBClass2Tester


(x_train, y_train), (x_test, y_test) = adult_utils.get_dataset_bin_relationship()

preprocessor = adult_utils.get_dataset_preprocessor(x_train)

x_train_encoded = preprocessor.fit_transform(x_train)
x_test_encoded = preprocessor.transform(x_test)

k_trials = (50, 100, 200, 400, 800, 1000)
QIs = [['workclass', 'education-num', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'native-country'],
      ['age', 'workclass', 'education-num', 'marital-status', 'occupation', 'relationship', 'race', 'sex',
       'hours-per-week', 'native-country'],
      ['age', 'workclass', 'education-num', 'marital-status', 'occupation', 'relationship', 'race', 'sex',
       'capital-gain', 'capital-loss', 'hours-per-week', 'native-country']]

tester = AnonymGBClass2Tester(attack_column='relationship', sensitive_column='relationship',
                              input_veclen=x_train_encoded.shape[1], sample_len=len(x_train))

acc_vanilla, acc_proc_list = tester.perform_anonymtest_QIs(x_train, x_train_encoded, y_train, x_test_encoded, y_test,
                                                           preprocessor, QIs, k_trials)

plot_path = 'results/plots/'
yminmax_list = [[0.6, 0.9], [0.55, 1.0], [0.55, 1.0], [0.55, 1.0], [0.55, 1.0]]

def save_results(QI, acc_proc):
    fname_base = 'anonymization_adult-NN2-Qi' + str(len(QI))
    imfname = fname_base + '.png'

    results_utils.save_results([plot_path+'inference/'+imfname,
                                plot_path+'member_attack/'+imfname,
                                plot_path+'attrb_black_attack/'+imfname],
                               k_trials, acc_proc, acc_vanilla, 'results/data/'+fname_base+'.npy', yminmax_list)

for QI, acc_proc in zip(QIs, acc_proc_list):
    save_results(QI, acc_proc)
