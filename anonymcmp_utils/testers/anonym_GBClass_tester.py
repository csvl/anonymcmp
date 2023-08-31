from tensorflow import keras
from anonymcmp_utils.nnGoldsteenModels import GoldsteenBClassModel1, GoldsteenBClassModel2
from .anonym_NN_tester import AnonymNNTester


class AnonymGBClass1Tester(AnonymNNTester):
    def __init__(self, attack_column, sensitive_column):
        super(AnonymGBClass1Tester, self).__init__(attack_column, sensitive_column)

    def get_trained_model(self, x, y):
        model = GoldsteenBClassModel1(keras.layers.Input(shape=(x.shape[1],)))
        model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])
        model.fit(x, y.to_numpy(), epochs=100)

        return model

    def get_prediction_accuracy(self, optmodel, x_test_encoded, y_test):
        _, pred_acc = optmodel.evaluate(x_test_encoded, y_test.to_numpy(), verbose=0)

        return pred_acc


    def get_predictions(self, model, x):
        return (model.predict(x) >= 0.5).astype('int')


class AnonymGBClass2Tester(AnonymNNTester):
    def __init__(self, attack_column, sensitive_column):
        super(AnonymGBClass2Tester, self).__init__(attack_column, sensitive_column)

    def get_trained_model(self, x, y):
        model = GoldsteenBClassModel2(keras.layers.Input(shape=(x.shape[1],)))
        model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.0001), loss='binary_crossentropy', metrics=['accuracy'])
        model.fit(x, y.to_numpy(), epochs=100)

        return model

    def get_prediction_accuracy(self, optmodel, x_test_encoded, y_test):
        _, pred_acc = optmodel.evaluate(x_test_encoded, y_test.to_numpy(), verbose=0)

        return pred_acc

    def get_predictions(self, model, x):
        return (model.predict(x) >= 0.5).astype('int')
