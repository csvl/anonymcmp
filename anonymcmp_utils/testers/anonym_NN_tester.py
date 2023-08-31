from art.estimators.classification.tensorflow import TensorFlowV2Classifier
from .anonym_tester import AnonymTester

class AnonymNNTester(AnonymTester):
    def __init__(self, attack_column, sensitive_column):
        super(AnonymNNTester, self).__init__(attack_column, sensitive_column)

    def get_trained_model(self, x, y):
        pass

    def get_prediction_accuracy(self, optmodel, x_test_encoded, y_test):
        pass

    def get_predictions(self, model, x):
        pass

    def get_art_classifier(self, optmodel, x_train_encoded, y_train):
        return TensorFlowV2Classifier(optmodel, len(y_train.unique()), (x_train_encoded.shape[1],), optmodel.loss,
                                      optmodel.optimizer)

    def isNNSubClass(selfs):
        return True