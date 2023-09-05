from art.estimators.classification.tensorflow import TensorFlowV2Classifier
import dp_accounting
import numpy as np
from scipy import interpolate
from tensorflow_privacy.privacy.optimizers.dp_optimizer_keras import DPKerasAdamOptimizer
from .anonym_tester import AnonymTester

class AnonymNNTester(AnonymTester):
    def __init__(self, attack_column, sensitive_column, input_veclen, sample_len, learning_rate, loss, epochs=100, batch_size=32):
        super(AnonymNNTester, self).__init__(attack_column, sensitive_column)

        self.input_veclen = input_veclen
        self.learning_rate = learning_rate
        self.loss = loss
        self.epochs = epochs

        noise_multipliers = np.linspace(0.1, 1.0).tolist() + [pow(10, i) for i in range(1, 4)]
        epsilons = [self.compute_epsilon(epochs*sample_len//batch_size, nm, sample_len, batch_size)
                    for nm in noise_multipliers]

        self.f = interpolate.interp1d(np.log(epsilons), np.log(noise_multipliers), kind='slinear', fill_value='extrapolate')

    def get_trained_model(self, x, y):
        pass

    def get_prediction_accuracy(self, optmodel, x_test_encoded, y_test):
        pass

    def get_art_classifier(self, optmodel, x_train_encoded, y_train):
        return TensorFlowV2Classifier(optmodel, len(y_train.unique()), (x_train_encoded.shape[1],), optmodel.loss,
                                      optmodel.optimizer)
    def get_predictions(self, model, x):
        pass

    def measure_difpriv_accuracies(self, eps, categorical_features, x_train_encoded, preprocessor, y_train,
                                   x_test_encoded, y_test):
        pass

    def get_diffpriv_classifier(self, eps):
        nm = self.compute_noise_multiplier(eps)
        assert (nm != np.inf)

        model = self.get_model(self.input_veclen)
        model.compile(optimizer=DPKerasAdamOptimizer(learning_rate=self.learning_rate, l2_norm_clip=1.0,
                                                     noise_multiplier=nm, num_microbatches=1),
                      loss=self.loss, metrics=['accuracy'])
        return model

    def isNNSubClass(selfs):
        return True

    def compute_noise_multiplier(self, eps):
        return np.e**self.f(np.log(eps))

    def compute_epsilon(self, steps, nm, sample_len, batch_size):
        if nm == 0.0:
            return float('inf')
        orders = [1 + x / 10. for x in range(1, 100)] + list(range(12, 64))
        accountant = dp_accounting.rdp.RdpAccountant(orders)

        sampling_probability = batch_size / sample_len
        event = dp_accounting.SelfComposedDpEvent(
            dp_accounting.PoissonSampledDpEvent(sampling_probability, dp_accounting.GaussianDpEvent(nm)), steps)

        accountant.compose(event)

        return accountant.get_epsilon(target_delta=1 / pow(10, int(np.ceil(np.log10(sample_len)))))

