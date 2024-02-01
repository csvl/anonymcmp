from tensorflow import keras
from tensorflow.keras.utils import to_categorical
from anonymcmp_utils.nnGoldsteenModels import Goldsteen4ClassModel
import numpy as np
from .anonym_NN_tester import AnonymNNTester


class AnonymG4ClassTester(AnonymNNTester):
    def __init__(self, attack_column, sensitive_column, input_veclen, sample_len, learning_rate=0.001, epochs=100):
        super(AnonymG4ClassTester, self).__init__(attack_column, sensitive_column, input_veclen, sample_len,
                                                  learning_rate, 'categorical_crossentropy', epochs)

    def get_model(self, invec_size):
        return Goldsteen4ClassModel(keras.layers.Input(shape=(invec_size,)))

    def get_trained_model(self, x, y):
        assert (self.input_veclen == x.shape[1])
        model = self.get_model(x.shape[1])
        model.compile(optimizer=keras.optimizers.Adam(learning_rate=self.learning_rate), loss=self.loss,
                      metrics=['accuracy'])
        model.fit(x, to_categorical(y.to_numpy()), epochs=self.epochs, verbose=0)
        return model

    def get_prediction_accuracy(self, optmodel, x_test_encoded, y_test):
        pred_loss, pred_acc = optmodel.evaluate(x_test_encoded, to_categorical(y_test.to_numpy()), verbose=0)
        assert (not np.isnan(pred_loss))

        return pred_acc

    def get_predictions(self, model, x):
        return model.predict(x).argmax(axis=1).reshape(-1,1)

    def measure_difpriv_accuracies(self, eps, categorical_features, x_train_encoded, preprocessor, y_train,
                                   x_test_encoded, y_test):
        dp_clf = self.get_diffpriv_classifier(eps)
        dp_clf.fit(x_train_encoded, to_categorical(y_train.to_numpy()), epochs=self.epochs)

        return self.measure_accuracies(dp_clf, x_train_encoded, y_train, x_test_encoded, y_test, preprocessor,
                                       categorical_features)

