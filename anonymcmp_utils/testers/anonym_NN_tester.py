from art.estimators.classification.tensorflow import TensorFlowV2Classifier
import dp_accounting
import numpy as np
from scipy import interpolate
from tensorflow_privacy.privacy.optimizers.dp_optimizer_keras import DPKerasAdamOptimizer
from tensorflow_privacy.privacy.analysis.compute_dp_sgd_privacy import compute_dp_sgd_privacy
from tensorflow import keras
import tensorflow as tf
from .anonym_tester import AnonymTester

class AnonymNNTester(AnonymTester):
    def __init__(self, attack_column, sensitive_column, input_veclen, sample_len, learning_rate, loss, metrics,
                 bufsiz, epochs=100, batch_size=32):
        super(AnonymNNTester, self).__init__(attack_column, sensitive_column)

        self.input_veclen = input_veclen
        self.learning_rate = learning_rate
        self.loss = loss
        self.METRICS = metrics
        self.epochs = epochs
        self.BUFFER_SIZE = bufsiz

        noise_multipliers = np.linspace(0.1, 1.0).tolist() + [pow(10, i) for i in range(1, 4)]

        epsilons = [compute_dp_sgd_privacy(n=sample_len, batch_size=batch_size, noise_multiplier=nm, epochs=epochs,
                                           delta=1 / (sample_len*sample_len))[0]
                    for nm in noise_multipliers]

        self.f = interpolate.interp1d(np.log(epsilons), np.log(noise_multipliers), kind='slinear', fill_value='extrapolate')

    def get_trained_model(self, x, y, fname=None):
        pass

    def load_trained_model(self, fname):
        return keras.models.load_model(fname)

    def get_prediction_result(self, optmodel, x_test_encoded, y_test):
        pass

    def get_art_classifier(self, optmodel, x_train_encoded, y_train):
        return TensorFlowV2Classifier(optmodel, len(y_train.unique()), (x_train_encoded.shape[1],), optmodel.loss,
                                      optmodel.optimizer)
    def get_predictions(self, model, x):
        pass

    def measure_difpriv_accuracies(self, eps, categorical_features, x_train_encoded, preprocessor, y_train,
                                   x_test_encoded, y_test, model_path):
        pass

    def get_diffpriv_classifier(self, eps):
        nm = self.compute_noise_multiplier(eps)
        assert (nm != np.inf)

        model = self.get_model(self.input_veclen)
        model.compile(optimizer=DPKerasAdamOptimizer(learning_rate=self.learning_rate, l2_norm_clip=1.0,
                                                     noise_multiplier=nm, num_microbatches=1),
                      loss=self.loss, metrics=self.METRICS)
        return model

    def save_dpmodel(self, model, fname):
        model.save_weights(fname)

    def get_model(self, invec_size):
        pass

    def isNNSubClass(selfs):
        return True

    def compute_noise_multiplier(self, eps):
        return np.e**self.f(np.log(eps))

    def make_ds(self, features, labels):
        ds = tf.data.Dataset.from_tensor_slices((features, labels))  # .cache()
        ds = ds.shuffle(self.BUFFER_SIZE).repeat()
        return ds
