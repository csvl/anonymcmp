from tensorflow import keras
from anonymcmp_utils.nnGoldsteenModels import GoldsteenBClassModel1, GoldsteenBClassModel2
import numpy as np
from sklearn.metrics import classification_report, roc_auc_score
import tensorflow as tf
from sklearn.model_selection import train_test_split
from .anonym_NN_tester import AnonymNNTester


class AnonymGBClassTester(AnonymNNTester):
    def __init__(self, attack_column, sensitive_column, input_veclen, sample_len, learning_rate, epochs=100,
                 bufsiz=100000, batsiz=2048):
        super(AnonymGBClassTester, self).__init__(attack_column, sensitive_column, input_veclen, sample_len,
                                                  learning_rate, 'binary_crossentropy',
                                                  [keras.metrics.BinaryAccuracy(name='accuracy'),
                                                   #keras.metrics.AUC(name='prc', curve='PR')],
                                                   keras.metrics.AUC(name='auc')],
                                                  bufsiz, epochs)

        self.BATCH_SIZE = batsiz

    def get_model(self, invec_size):
        pass

    def get_trained_model(self, x, y, fname=None):
        assert (self.input_veclen == x.shape[1])
        model = self.get_model(x.shape[1])

        model.compile(optimizer=keras.optimizers.Adam(learning_rate=self.learning_rate), loss=self.loss,
                      metrics=self.METRICS)

        resampled_ds, val_ds, resampled_steps_per_epoch = self.make_balanced_ds(x, y)

        model.fit(resampled_ds, epochs=self.epochs, steps_per_epoch=resampled_steps_per_epoch,
                  callbacks=[self.early_stopping()], validation_data=val_ds)

        if fname is not None:
            model.save(fname)

        return model

    def make_balanced_ds(self, x, y):
        x_train, x_valid, y_train, y_valid = train_test_split(x, y, test_size=0.2)

        bool_train_labels = y_train == 1

        pos_features = x_train[bool_train_labels]
        neg_features = x_train[~bool_train_labels]

        pos_labels = y_train[bool_train_labels]
        neg_labels = y_train[~bool_train_labels]

        pos_ds = self.make_ds(pos_features, pos_labels)
        neg_ds = self.make_ds(neg_features, neg_labels)

        resampled_ds = tf.data.Dataset.sample_from_datasets([pos_ds, neg_ds], weights=[0.5, 0.5])
        resampled_ds = resampled_ds.batch(self.BATCH_SIZE).prefetch(2)

        resampled_steps_per_epoch = int(np.ceil(2.0 * min(sum(bool_train_labels), sum(y_train == 0)) / self.BATCH_SIZE))

        val_ds = tf.data.Dataset.from_tensor_slices((x_valid, y_valid)).cache()
        val_ds = val_ds.batch(self.BATCH_SIZE).prefetch(2)

        return  resampled_ds, val_ds, resampled_steps_per_epoch


    def early_stopping(self):
        return keras.callbacks.EarlyStopping(
            #monitor='val_prc',
            monitor='val_auc',
            verbose=1,
            patience=10,
            mode='max',
            restore_best_weights=True)

    def get_prediction_result(self, optmodel, x_test_encoded, y_test):
        pred_loss, pred_acc, _ = optmodel.evaluate(x_test_encoded, y_test.to_numpy(), verbose=0)
        assert (not np.isnan(pred_loss))

        prediction = optmodel.predict(x_test_encoded)
        report = classification_report(y_test, prediction>0.5, labels=[1], digits=3, output_dict=True)['1']
        auc_score = roc_auc_score(y_test, prediction)

        return [pred_acc, report['precision'], report['recall'], report['f1-score'], auc_score]

    def get_predictions(self, model, x):
        return (model.predict(x) >= 0.5).astype('int')

    def measure_difpriv_accuracies(self, eps, categorical_features, x_train_encoded, preprocessor, y_train,
                                   x_test_encoded, y_test, model_path=None):
        dp_clf = self.get_diffpriv_classifier(eps)
        # the following cause error while training
        #resampled_ds, val_ds, resampled_steps_per_epoch = self.make_balanced_ds(x_train_encoded, y_train)
        #dp_clf.fit(resampled_ds, epochs=self.epochs, steps_per_epoch=resampled_steps_per_epoch,
        #          callbacks=[self.early_stopping()], validation_data=val_ds) #, verbose=0)

        dp_clf.fit(x_train_encoded, y_train, epochs=self.epochs, verbose=0)

        if model_path is not None:
            self.save_dpmodel(dp_clf, model_path)

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
