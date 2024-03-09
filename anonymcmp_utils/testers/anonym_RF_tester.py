import numpy as np
from sklearn.ensemble import RandomForestClassifier
from art.estimators.classification.scikitlearn import ScikitlearnRandomForestClassifier
import diffprivlib.models as dp
from joblib import dump, load
from .anonym_tester import AnonymTester

class AnonymRFTester(AnonymTester):
    def __init__(self, attack_column, sensitive_column, max_depth):
        super(AnonymRFTester, self).__init__(attack_column, sensitive_column)
        self.max_depth = max_depth

    def get_trained_model(self, x, y, fname=None):
        model = RandomForestClassifier(max_depth=self.max_depth)
        model = model.fit(x, y)

        if fname is not None:
            dump(model, fname)

        return model

    def get_prediction_accuracy(self, optmodel, x_test_encoded, y_test):
        return optmodel.score(x_test_encoded, y_test)

    def get_art_classifier(self, optmodel, x_train_encoded, y_train):
        return ScikitlearnRandomForestClassifier(optmodel)

    def get_predictions(self, model, x):
        return np.array(
            [np.argmax(arr) for arr in ScikitlearnRandomForestClassifier(model).predict(x)]).reshape(-1, 1)

    def get_diffpriv_classifier(self, eps):
        return dp.RandomForestClassifier(epsilon=eps, data_norm=5, max_depth=self.max_depth)

    def save_dpmodel(self, model, fname):
        dump(model, fname)
