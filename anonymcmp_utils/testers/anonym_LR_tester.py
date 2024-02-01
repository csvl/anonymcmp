import numpy as np
from sklearn.linear_model import LogisticRegression
from art.estimators.classification.scikitlearn import ScikitlearnLogisticRegression
import diffprivlib.models as dp
from .anonym_tester import AnonymTester

class AnonymLRTester(AnonymTester):
    def __init__(self, attack_column, sensitive_column, max_iter):
        super(AnonymLRTester, self).__init__(attack_column, sensitive_column)
        self.max_iter = max_iter

    def get_trained_model(self, x, y):
        model = LogisticRegression(solver="lbfgs", max_iter=self.max_iter)
        model = model.fit(x, y)
        return model

    def get_prediction_accuracy(self, optmodel, x_test_encoded, y_test):
        return optmodel.score(x_test_encoded, y_test)

    def get_art_classifier(self, optmodel, x_train_encoded, y_train):
        return ScikitlearnLogisticRegression(optmodel)

    def get_predictions(self, model, x):
        return np.array(
            [np.argmax(arr) for arr in ScikitlearnLogisticRegression(model).predict(x)]).reshape(-1, 1)

    def get_diffpriv_classifier(self, eps):
        return dp.LogisticRegression(epsilon=eps, data_norm=5, max_iter=self.max_iter)
