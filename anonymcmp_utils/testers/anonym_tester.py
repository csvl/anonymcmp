import numpy as np
from apt.utils.datasets import ArrayDataset
from apt.anonymization import Anonymize
from mondrian import anonymize, MondrianOption
from anonymcmp_utils import measure_membership_attack_accuracy, measure_attribute_bbox_attack_accuracy, \
    measure_attribute_wboxLDT_attack_accuracy, measure_attribute_wboxDT_attack_accuracy


class AnonymTester:
    def __init__(self, attack_column, sensitive_column):
        self.attack_column = attack_column
        self.sensitive_column = sensitive_column

    def get_trained_model(self, x, y):
        pass

    def get_prediction_accuracy(self):
        pass

    def get_art_classifier(self, optmodel, x_train_encoded, y_train):
        pass

    def get_predictions(self, model, x):
        pass

    def get_diffpriv_classifier(self, eps):
        pass

    def isNNSubClass(selfs):
        return False

    def perform_test(self, x_train, x_train_encoded, y_train, x_test_encoded, y_test, preprocessor, QI, k_trials,
                     epsilons=None, numtest=10, multitest_vanilla=False):

        categorical_features = x_train.select_dtypes(['object']).columns.to_list()

        list_accvanilla = []
        for i in range(numtest if multitest_vanilla else 1):
            print(i)
            vanilla_model = self.get_trained_model(x_train_encoded, y_train)
            acc_vanilla = self.measure_accuracies(vanilla_model, x_train_encoded, y_train, x_test_encoded, y_test,
                                                  preprocessor, categorical_features)
            list_accvanilla.append(acc_vanilla)

        acc_proc = self.measure_anonym_accuracies(x_train, categorical_features, QI, self.sensitive_column, preprocessor,
                                                  x_train_encoded, y_train, k_trials, vanilla_model, x_test_encoded,
                                                  y_test, numtest)

        if epsilons != None:
            acc_proc['Differential privacy'] = [
                [self.measure_difpriv_accuracies(eps, categorical_features, x_train_encoded, preprocessor, y_train,
                                                 x_test_encoded, y_test) for _ in range(numtest)]
                for eps in epsilons]

        return np.array(list_accvanilla), acc_proc


    def perform_anonymtest_QIs(self, x_train, x_train_encoded, y_train, x_test_encoded, y_test, preprocessor, QIs,
                               k_trials, numtest=10):
        vanilla_model = self.get_trained_model(x_train_encoded, y_train)

        categorical_features = x_train.select_dtypes(['object']).columns.to_list()

        acc_vanilla = self.measure_accuracies(vanilla_model, x_train_encoded, y_train, x_test_encoded, y_test,
                                              preprocessor, categorical_features)

        acc_proc_list = [self.measure_anonym_accuracies(x_train, categorical_features, QI, self.sensitive_column,
                                                        preprocessor, x_train_encoded, y_train, k_trials, vanilla_model,
                                                        x_test_encoded, y_test, numtest)
                         for QI in QIs]

        return np.array([acc_vanilla]), acc_proc_list

    def measure_accuracies(self, optmodel, x_train_encoded, y_train, x_test_encoded, y_test, preprocessor, categorical_features):
        attack_feature = self.get_attack_idx(preprocessor, self.attack_column, categorical_features)

        x_train_for_attack, x_train_feature, values, priors = self.get_attribute_measure_params(x_train_encoded, attack_feature)

        # use half of each dataset for training the attack
        attack_train_ratio = 0.5
        attack_train_size = int(len(x_train_encoded) * attack_train_ratio)
        attack_test_size = int(len(x_test_encoded) * attack_train_ratio)

        pred_acc = self.get_prediction_accuracy(optmodel, x_test_encoded, y_test)

        classifier = self.get_art_classifier(optmodel, x_train_encoded, y_train)

        # x_train_encoded.astype(np.float32), x_test_encoded.astype(np.float32)
        mmb_acc = measure_membership_attack_accuracy(classifier, x_train_encoded, y_train,
                                                     x_test_encoded, y_test,
                                                     attack_train_size, attack_test_size)

        x_train_encoded_predictions = self.get_predictions(optmodel, x_train_encoded)

        #x_train_encoded.astype(np.float32)
        attrb_acc = measure_attribute_bbox_attack_accuracy(classifier, x_train_encoded,
                                                           attack_train_size,
                                                           x_train_encoded_predictions, attack_feature,
                                                           x_train_for_attack, values, x_train_feature)

        if type(self).__name__ != 'AnonymDTTester':
            return [pred_acc, mmb_acc, attrb_acc]

        attrlw_acc = measure_attribute_wboxLDT_attack_accuracy(classifier, x_train_encoded_predictions,
                                                               attack_feature,
                                                               x_train_for_attack, values, priors, x_train_feature)

        attrw_acc = measure_attribute_wboxDT_attack_accuracy(classifier, x_train_encoded_predictions,
                                                             attack_feature,
                                                             x_train_for_attack, values, priors, x_train_feature)

        return [pred_acc, mmb_acc, attrb_acc, attrlw_acc, attrw_acc]


    def measure_anonym_accuracies(self, x_train, categorical_features, QI, sensitive_column, preprocessor,
                                  x_train_encoded, y_train, k_trials, vanilla_model, x_test_encoded, y_test, numtest):
        x_train_predictions = self.get_predictions(vanilla_model, x_train_encoded)

        accuracies = {}
        accuracies['AG'] = [
            [self.measure_mlanoym_accuracies(k, QI, categorical_features, x_train, x_train_predictions, y_train,
                                            preprocessor, x_train_encoded, x_test_encoded, y_test)
             for _ in range(numtest)]
            for k in k_trials]

        accuracies['Mondrian'] = [
            [self.measure_mondrian_accuracies(k, MondrianOption.Non, x_train, categorical_features, QI, sensitive_column,
                                             preprocessor, x_train_encoded, y_train, x_test_encoded, y_test)
             for _ in range(numtest)]
            for k in k_trials]

        accuracies['l-diverse'] = [
            [self.measure_mondrian_accuracies(k, MondrianOption.ldiv, x_train, categorical_features, QI,
                                              sensitive_column, preprocessor, x_train_encoded, y_train, x_test_encoded,
                                              y_test, 2) for _ in range(numtest)]
            for k in k_trials]

        accuracies['t-closeness'] = [
            [self.measure_mondrian_accuracies(k, MondrianOption.tclose, x_train, categorical_features, QI,
                                              sensitive_column, preprocessor, x_train_encoded, y_train, x_test_encoded,
                                              y_test, 0.2) for _ in range(numtest)]
            for k in k_trials]

        return accuracies

    def measure_mlanoym_accuracies(self, k, QI, categorical_features, x_train, x_train_predictions, y_train, preprocessor,
                                   x_train_encoded, x_test_encoded, y_test):
        anonymizer = Anonymize(k, QI, categorical_features=categorical_features)
        anon = anonymizer.anonymize(ArrayDataset(x_train, x_train_predictions))
        anon = anon.astype(x_train.dtypes)

        anon_encoded = preprocessor.transform(anon)

        anon_model = self.get_trained_model(anon_encoded, y_train)

        return self.measure_accuracies(anon_model, x_train_encoded, y_train, x_test_encoded, y_test, preprocessor,
                                       categorical_features)


    def measure_mondrian_accuracies(self, k, option, x_train, categorical_features, QI, sensitive_column, preprocessor,
                                    x_train_encoded, y_train, x_test_encoded, y_test, l_or_p=1):
        anon = anonymize(x_train, set(categorical_features), QI, sensitive_column, k, option, l_or_p)
        anon = anon.astype(x_train.dtypes)

        anon_encoded = preprocessor.transform(anon)

        anon_model = self.get_trained_model(anon_encoded, y_train)

        return self.measure_accuracies(anon_model, x_train_encoded, y_train, x_test_encoded, y_test, preprocessor,
                                       categorical_features)


    def measure_difpriv_accuracies(self, eps, categorical_features, x_train_encoded, preprocessor, y_train,
                                   x_test_encoded, y_test):
        dp_clf = self.get_diffpriv_classifier(eps)
        dp_clf = dp_clf.fit(x_train_encoded, y_train)

        return self.measure_accuracies(dp_clf, x_train_encoded, y_train, x_test_encoded, y_test, preprocessor,
                                       categorical_features)


    def get_attack_idx(self, preprocessor, attack_column, categorical_features):
        hotencoder_feature_names = preprocessor.named_transformers_['cat'].named_steps['hotencoder'].get_feature_names(
            categorical_features)
        idx_attack_hotencoder = int(np.argwhere([attack_column in f for f in hotencoder_feature_names])[0][0])

        return len(preprocessor.transformers[0][-1]) + idx_attack_hotencoder


    def get_attribute_measure_params(self, x_train_encoded, attack_feature):
        # training data without attacked feature
        x_train_for_attack = np.delete(x_train_encoded, attack_feature, 1)
        # only attacked feature
        x_train_feature = x_train_encoded[:, attack_feature].copy()

        # get inferred values
        values = np.unique(x_train_feature)

        priors = [(x_train_feature == v).sum() / len(x_train_feature) for v in values]

        return x_train_for_attack, x_train_feature, values, priors
