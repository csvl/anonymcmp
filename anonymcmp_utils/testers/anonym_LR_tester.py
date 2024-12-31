import numpy as np
from sklearn.linear_model import LogisticRegression
from art.estimators.classification.scikitlearn import ScikitlearnLogisticRegression
import diffprivlib.models as dp
from joblib import dump, load
from sklearn.metrics import classification_report, roc_auc_score
from .anonym_tester import AnonymTester
from tensorflow.keras.utils import to_categorical

class AnonymLRTester(AnonymTester):
    def __init__(self, attack_column, sensitive_column, max_iter):
        super(AnonymLRTester, self).__init__(attack_column, sensitive_column)
        self.max_iter = max_iter

    def get_trained_model(self, x, y, fname=None):
        model = LogisticRegression(solver="lbfgs", max_iter=self.max_iter)
        model = model.fit(x, y)

        if fname is not None:
            dump(model, fname)

        return model

    def get_prediction_result(self, optmodel, x_test_encoded, y_test):
        pred_acc = optmodel.score(x_test_encoded, y_test)

        prediction = optmodel.predict(x_test_encoded)
        if len(np.unique(y_test)) == 2:  # binary class
            report = classification_report(y_test, prediction, labels=[1], digits=3, output_dict=True)['1']
            auc_score = roc_auc_score(y_test, prediction)
        else:
            report = classification_report(y_test, prediction, digits=3, output_dict=True)[
            'macro avg']
            auc_score = roc_auc_score(to_categorical(y_test.to_numpy()), to_categorical(prediction))

        return [pred_acc, report['precision'], report['recall'], report['f1-score'], auc_score]

    def get_art_classifier(self, optmodel, x_train_encoded, y_train):
        return ScikitlearnLogisticRegression(optmodel)

    def get_predictions(self, model, x):
        return np.array(
            [np.argmax(arr) for arr in ScikitlearnLogisticRegression(model).predict(x)]).reshape(-1, 1)

    def get_diffpriv_classifier(self, eps):
        return dp.LogisticRegression(epsilon=eps, data_norm=5, max_iter=self.max_iter)

    def save_dpmodel(self, model, fname):
        dump(model, fname)
