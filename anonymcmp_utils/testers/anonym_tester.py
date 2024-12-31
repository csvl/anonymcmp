import numpy as np
import os
from apt.utils.datasets import ArrayDataset
from apt.anonymization import Anonymize
from mondrian import anonymize, MondrianOption
from joblib import load
from anonymcmp_utils import measure_membership_attack, measure_attribute_bbox_attack_accuracy, \
    measure_attribute_wboxLDT_attack_accuracy, measure_attribute_wboxDT_attack_accuracy

class AnonymTester:
    def __init__(self, attack_column, sensitive_column):
        self.attack_column = attack_column
        self.sensitive_column = sensitive_column

    def get_trained_model(self, x, y, fname=None):
        pass

    def load_trained_model(self, fname):
        return load(fname)

    def get_prediction_result(self, optmodel, x_test_encoded, y_test):
        pass

    def get_art_classifier(self, optmodel, x_train_encoded, y_train):
        pass

    def get_predictions(self, model, x):
        pass

    def get_diffpriv_classifier(self, eps):
        pass

    def save_dpmodel(self, model, fname):
        pass

    def isNNSubClass(selfs):
        return False

    def perform_test(self, x_train, x_train_encoded, y_train, x_test_encoded, y_test, preprocessor, QI, k_trials,
                     npfname, epsilons=None, numtest=10, multitest_vanilla=False, model_path='results/models'):

        categorical_features = x_train.select_dtypes(['object']).columns.to_list()

        if not os.path.exists(model_path):
            os.mkdir(model_path)

        list_accvanilla = []
        for i in range(numtest if multitest_vanilla else 1):
            vanilla_model = self.get_trained_model(x_train_encoded, y_train, model_path + '/vanilla_%d' % i)
            #vanilla_model = self.load_trained_model(model_path + '/vanilla_%d' % i)
            acc_vanilla = self.measure_accuracies(vanilla_model, x_train_encoded, y_train, x_test_encoded, y_test,
                                                  preprocessor, categorical_features)
            list_accvanilla.append(acc_vanilla)

        acc_vanilla = np.array(list_accvanilla)
        np.save(npfname, (acc_vanilla, None), allow_pickle=True)

        acc_proc = self.measure_anonym_accuracies(x_train, categorical_features, QI, self.sensitive_column, preprocessor,
                                                  x_train_encoded, y_train, k_trials, vanilla_model, x_test_encoded,
                                                  y_test, numtest, npfname, model_path)

        if epsilons != None:
            acc_proc['Differential privacy'] = [
                [self.measure_difpriv_accuracies(eps, categorical_features, x_train_encoded, preprocessor, y_train,
                                                 x_test_encoded, y_test, model_path + '/DP_%d_%.2f' % (i, eps))
                 for i in range(numtest)]
                for eps in epsilons]

        return acc_vanilla, acc_proc

    def update_AGanonym(self, x_train, x_train_encoded, y_train, x_test_encoded, y_test, preprocessor, QI, k_trials,
                        fname, numtest=10, model_path='results/models'):

        res = np.load(fname, allow_pickle=True)

        categorical_features = x_train.select_dtypes(['object']).columns.to_list()

        if not os.path.exists(model_path):
            os.mkdir(model_path)

        vanilla_model_fpath = model_path + '/vanilla_0'
        vanilla_model = self.load_trained_model(vanilla_model_fpath) if os.path.exists(vanilla_model_fpath) \
            else self.get_trained_model(x_train_encoded, y_train, vanilla_model_fpath)

        x_train_predictions = self.get_predictions(vanilla_model, x_train_encoded)

        acc_proc = res[1]

        acc_proc['AG'] = [
            [self.measure_mlanoym_accuracies(k, QI, categorical_features, x_train, x_train_predictions, y_train,
                                             preprocessor, x_train_encoded, x_test_encoded, y_test,
                                             model_path + '/AG_%d_%d' % (i, k)) for i in range(numtest)]
            for k in k_trials]

        return res[0], acc_proc

    def update_anonym(self, x_train, x_train_encoded, y_train, x_test_encoded, y_test, preprocessor, QI, k_trials,
                      npfname, numtest=10, model_path='results/models'):

        res = np.load(npfname, allow_pickle=True)

        categorical_features = x_train.select_dtypes(['object']).columns.to_list()

        if not os.path.exists(model_path):
            os.mkdir(model_path)

        vanilla_model_fpath = model_path + '/vanilla_0'
        vanilla_model = self.load_trained_model(vanilla_model_fpath) if os.path.exists(vanilla_model_fpath) \
            else self.get_trained_model(x_train_encoded, y_train, vanilla_model_fpath)

        acc_proc = self.measure_anonym_accuracies(x_train, categorical_features, QI, self.sensitive_column,
                                                  preprocessor,
                                                  x_train_encoded, y_train, k_trials, vanilla_model, x_test_encoded,
                                                  y_test, numtest, npfname, model_path)

        acc_proc['Differential privacy'] = res[1]['Differential privacy']

        return res[0], acc_proc


    def update_dp(self, x_train, x_train_encoded, y_train, x_test_encoded, y_test, preprocessor, fname, epsilons,
                  numtest=10, model_path=None):

        res = np.load(fname, allow_pickle=True)
        acc_proc = res[1]

        if model_path is not None and not os.path.exists(model_path):
            os.mkdir(model_path)

        categorical_features = x_train.select_dtypes(['object']).columns.to_list()

        acc_proc['Differential privacy'] = [
            [self.measure_difpriv_accuracies(eps, categorical_features, x_train_encoded, preprocessor, y_train,
                                             x_test_encoded, y_test, model_path + '/DP_%d_%.2f' % (i, eps))
             for i in range(numtest)]
            for eps in epsilons]

        return res[0], acc_proc

    def perform_anonymtest_QIs(self, x_train, x_train_encoded, y_train, x_test_encoded, y_test, preprocessor, QIs,
                               k_trials, vanilla_model_fpath, npbasefname, numtest=10, model_root_path='results/models'):

        vanilla_model = self.load_trained_model(vanilla_model_fpath)

        categorical_features = x_train.select_dtypes(['object']).columns.to_list()

        acc_vanilla = self.measure_accuracies(vanilla_model, x_train_encoded, y_train, x_test_encoded, y_test,
                                              preprocessor, categorical_features)

        acc_proc_list = [self.measure_anonym_accuracies(x_train, categorical_features, QI, self.sensitive_column,
                                                        preprocessor, x_train_encoded, y_train, k_trials, vanilla_model,
                                                        x_test_encoded, y_test, numtest, npbasefname+str(len(QI))+'.npy',
                                                        model_root_path+str(len(QI)))
                         for QI in QIs]

        return np.array([acc_vanilla]), acc_proc_list

    def measure_accuracies(self, optmodel, x_train_encoded, y_train, x_test_encoded, y_test, preprocessor, categorical_features):

        attack_feature = self.get_attack_idx(preprocessor, self.attack_column, categorical_features)

        x_train_for_attack, x_train_feature, values, priors = self.get_attribute_measure_params(x_train_encoded, attack_feature)

        pred_res = self.get_prediction_result(optmodel, x_test_encoded, y_test)

        classifier = self.get_art_classifier(optmodel, x_train_encoded, y_train)

        # use half of balanced each of dataset for training the attack:
        attack_train_ratio = 0.5
        attack_train_size = int(min(len(y_train), len(y_test)) * attack_train_ratio)
        attack_test_size = int(min(len(y_train), len(y_test))* (1.0 - attack_train_ratio))
        
        # x_train_encoded.astype(np.float32), x_test_encoded.astype(np.float32)
        mmb_res = measure_membership_attack(classifier, x_train_encoded, y_train, x_test_encoded, y_test,
                                            attack_train_size, attack_test_size)

        return [pred_res, mmb_res]

        """
        x_train_encoded_predictions = self.get_predictions(optmodel, x_train_encoded)

        # x_train_encoded.astype(np.float32)
        attrb_acc = measure_attribute_bbox_attack_accuracy(classifier, x_train_encoded,
                                                           attack_train_size,
                                                           x_train_encoded_predictions, attack_feature,
                                                           x_train_for_attack, values, x_train_feature)

        if type(self).__name__ != 'AnonymDTTester':
            return [pred_res, mmb_res, attrb_acc]

        attrlw_acc = measure_attribute_wboxLDT_attack_accuracy(classifier, x_train_encoded_predictions,
                                                               attack_feature,
                                                               x_train_for_attack, values, priors, x_train_feature)

        attrw_acc = measure_attribute_wboxDT_attack_accuracy(classifier, x_train_encoded_predictions,
                                                             attack_feature,
                                                             x_train_for_attack, values, priors, x_train_feature)

        return [pred_res, mmb_res, attrb_acc, attrlw_acc, attrw_acc]
        """

    def measure_anonym_accuracies(self, x_train, categorical_features, QI, sensitive_column, preprocessor,
                                  x_train_encoded, y_train, k_trials, vanilla_model, x_test_encoded, y_test, numtest,
                                  npfname, model_path):

        x_train_predictions = self.get_predictions(vanilla_model, x_train_encoded)

        acc_vanilla, _ = np.load(npfname, allow_pickle=True)

        accuracies = {}
        accuracies['AG'] = np.array([
            [self.measure_mlanoym_accuracies(k, QI, categorical_features, x_train, x_train_predictions, y_train,
                                             preprocessor, x_train_encoded, x_test_encoded, y_test,
                                             model_path + '/AG_%d_%d' % (i, k)) for i in range(numtest)]
            for k in k_trials])

        np.save(npfname, (acc_vanilla, accuracies), allow_pickle=True)

        accuracies['Mondrian'] = np.array([
            [self.measure_mondrian_accuracies(k, MondrianOption.Non, x_train, categorical_features, QI,
                                              sensitive_column,
                                              preprocessor, x_train_encoded, y_train, x_test_encoded, y_test,
                                              model_fname=model_path + '/Mondrian_%d_%d' % (i, k)) for i in
             range(numtest)]
            for k in k_trials])

        np.save(npfname, (acc_vanilla, accuracies), allow_pickle=True)

        accuracies['l-diverse'] = np.array([
            [self.measure_mondrian_accuracies(k, MondrianOption.ldiv, x_train, categorical_features, QI,
                                              sensitive_column, preprocessor, x_train_encoded, y_train, x_test_encoded,
                                              y_test, 2, model_path + '/l-diverse_%d_%d' % (i, k)) for i in
             range(numtest)]
            for k in k_trials])

        np.save(npfname, (acc_vanilla, accuracies), allow_pickle=True)

        accuracies['t-closeness'] = np.array([
            [self.measure_mondrian_accuracies(k, MondrianOption.tclose, x_train, categorical_features, QI,
                                              sensitive_column, preprocessor, x_train_encoded, y_train, x_test_encoded,
                                              y_test, 0.2, model_path + '/t-closeness_%d_%d' % (i, k))
             for i in range(numtest)]
            for k in k_trials])

        np.save(npfname, (acc_vanilla, accuracies), allow_pickle=True)

        return accuracies

    def measure_mlanoym_accuracies(self, k, QI, categorical_features, x_train, x_train_predictions, y_train,
                                   preprocessor,
                                   x_train_encoded, x_test_encoded, y_test, model_fname):

        if os.path.exists(model_fname):
            anon_model = self.load_trained_model(model_fname)
        else:
            anonymizer = Anonymize(k, QI, categorical_features=categorical_features)
            anon = anonymizer.anonymize(ArrayDataset(x_train, x_train_predictions))
            anon = anon.astype(x_train.dtypes)

            anon_encoded = preprocessor.transform(anon)

            anon_model = self.get_trained_model(anon_encoded, y_train, model_fname)

        return self.measure_accuracies(anon_model, x_train_encoded, y_train, x_test_encoded, y_test, preprocessor,
                                       categorical_features)

    def measure_mondrian_accuracies(self, k, option, x_train, categorical_features, QI, sensitive_column, preprocessor,
                                    x_train_encoded, y_train, x_test_encoded, y_test, l_or_p=1, model_fname=None):

        if os.path.exists(model_fname):
            anon_model = self.load_trained_model(model_fname)
        else:
            anon = anonymize(x_train, set(categorical_features), QI, sensitive_column, k, option, l_or_p)
            anon = anon.astype(x_train.dtypes)

            anon_encoded = preprocessor.transform(anon)

            anon_model = self.get_trained_model(anon_encoded, y_train, model_fname)

        return self.measure_accuracies(anon_model, x_train_encoded, y_train, x_test_encoded, y_test, preprocessor,
                                       categorical_features)

    def measure_difpriv_accuracies(self, eps, categorical_features, x_train_encoded, preprocessor, y_train,
                                   x_test_encoded, y_test, model_path=None):

        if os.path.exists(model_path):
            dp_clf = self.load_trained_model(model_path) #not working for NN
        else:
            dp_clf = self.get_diffpriv_classifier(eps)
            dp_clf = dp_clf.fit(x_train_encoded, y_train)

            if model_path is not None:
                self.save_dpmodel(dp_clf, model_path)

        return self.measure_accuracies(dp_clf, x_train_encoded, y_train, x_test_encoded, y_test, preprocessor,
                                       categorical_features)

    def get_attack_idx(self, preprocessor, attack_column, categorical_features):
        hotencoder_feature_names = preprocessor.named_transformers_['cat'].named_steps[
            'hotencoder'].get_feature_names_out(categorical_features)
        idx_attack_hotencoder = int(np.argwhere([attack_column in f for f in hotencoder_feature_names])[0][0])

        return len(preprocessor.transformers[0][-1]) + idx_attack_hotencoder

    def get_attribute_measure_params(self, x_train_encoded, attack_feature):
        # training data without attacked feature
        x_train_for_attack = np.delete(x_train_encoded, attack_feature, 1)
        # only attacked feature
        x_train_feature = x_train_encoded[:, attack_feature].copy()

        # get inferred values
        values = list(np.unique(x_train_feature)) if self.isNNSubClass() else list(
            np.unique(x_train_feature).astype('int'))

        priors = [(x_train_feature == v).sum() / len(x_train_feature) for v in values]

        return x_train_for_attack, x_train_feature, values, priors
