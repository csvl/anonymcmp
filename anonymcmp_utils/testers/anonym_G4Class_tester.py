from tensorflow import keras
from tensorflow.keras.utils import to_categorical
from anonymcmp_utils.nnGoldsteenModels import Goldsteen4ClassModel
from .anonym_NN_tester import AnonymNNTester


class AnonymG4ClassTester(AnonymNNTester):
    def __init__(self, attack_column, sensitive_column):
        super(AnonymG4ClassTester, self).__init__(attack_column, sensitive_column)

    def get_trained_model(self, x, y):
        model = Goldsteen4ClassModel(keras.layers.Input(shape=(x.shape[1],)))
        model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
        model.fit(x, to_categorical(y.to_numpy()), epochs=100)

        return model

    def get_prediction_accuracy(self, optmodel, x_test_encoded, y_test):
        _, pred_acc = optmodel.evaluate(x_test_encoded, to_categorical(y_test.to_numpy()), verbose=0)

        return pred_acc

    def get_predictions(self, model, x):
        return model.predict(x).argmax(axis=1).reshape(-1,1)
