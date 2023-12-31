import numpy as np
from sklearn.tree import DecisionTreeClassifier
from art.estimators.classification.scikitlearn import ScikitlearnDecisionTreeClassifier
import diffprivlib.models as dp
from .anonym_tester import AnonymTester

class AnonymDTTester(AnonymTester):
    def __init__(self, attack_column, sensitive_column):
        super(AnonymDTTester, self).__init__(attack_column, sensitive_column)


    def get_trained_model(self, x, y):
        model = DecisionTreeClassifier()
        model = model.fit(x, y)

        return model


    def get_prediction_accuracy(self, optmodel, x_test_encoded, y_test):
        pred_acc = optmodel.score(x_test_encoded, y_test)

        return pred_acc


    def get_art_classifier(self, optmodel, x_train_encoded, y_train):
        return ScikitlearnDecisionTreeClassifier(optmodel)


    def get_predictions(self, model, x):
        return np.array(
            [np.argmax(arr) for arr in ScikitlearnDecisionTreeClassifier(model).predict(x)]).reshape(-1, 1)

    def get_diffpriv_classifier(self, eps):
        return dp.DecisionTreeClassifier(epsilon=eps)
