from tensorflow import keras
from anonymcmp_utils.nnGoldsteenModels import GoldsteenBClassModel1, GoldsteenBClassModel2
import numpy as np
from .anonym_NN_tester import AnonymNNTester


class AnonymGBClassTester(AnonymNNTester):
    def __init__(self, attack_column, sensitive_column, input_veclen, sample_len, learning_rate, epochs=100):
        super(AnonymGBClassTester, self).__init__(attack_column, sensitive_column, input_veclen, sample_len,
                                                  learning_rate, 'binary_crossentropy', epochs)

    def get_model(self, invec_size):
        pass

    def get_trained_model(self, x, y):
        assert (self.input_veclen == x.shape[1])
        model = self.get_model(x.shape[1])
        model.compile(optimizer=keras.optimizers.Adam(learning_rate=self.learning_rate), loss=self.loss,
                      metrics=['accuracy'])
        model.fit(x, y.to_numpy(), epochs=self.epochs, verbose=0)

        return model

    def get_prediction_accuracy(self, optmodel, x_test_encoded, y_test):
        pred_loss, pred_acc = optmodel.evaluate(x_test_encoded, y_test.to_numpy(), verbose=0)
        assert (not np.isnan(pred_loss))

        return pred_acc

    def get_predictions(self, model, x):
        return (model.predict(x) >= 0.5).astype('int')

    def measure_difpriv_accuracies(self, eps, categorical_features, x_train_encoded, preprocessor, y_train,
                                   x_test_encoded, y_test):
        dp_clf = self.get_diffpriv_classifier(eps)
        dp_clf.fit(x_train_encoded, y_train, epochs=self.epochs)

        return self.measure_accuracies(dp_clf, x_train_encoded, y_train, x_test_encoded, y_test, preprocessor,
                                       categorical_features)


class AnonymGBClass1Tester(AnonymGBClassTester):
    def __init__(self, attack_column, sensitive_column, input_veclen, sample_len, learning_rate=0.001, epochs=100):
        super(AnonymGBClass1Tester, self).__init__(attack_column, sensitive_column, input_veclen, sample_len,
                                                   learning_rate, epochs)

    def get_model(self, invec_size):
        return GoldsteenBClassModel1(keras.layers.Input(shape=(invec_size,)))


class AnonymGBClass2Tester(AnonymGBClassTester):
    def __init__(self, attack_column, sensitive_column, input_veclen, sample_len, learning_rate=0.0001, epochs=100):
        super(AnonymGBClass2Tester, self).__init__(attack_column, sensitive_column, input_veclen, sample_len,
                                                   learning_rate, epochs)

    def get_model(self, invec_size):
        return GoldsteenBClassModel2(keras.layers.Input(shape=(invec_size,)))
