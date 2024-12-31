from tensorflow import keras
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from anonymcmp_utils.nnGoldsteenModels import Goldsteen4ClassModel
import numpy as np
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import train_test_split
from .anonym_NN_tester import AnonymNNTester


class AnonymG4ClassTester(AnonymNNTester):
    def __init__(self, attack_column, sensitive_column, input_veclen, sample_len, learning_rate=0.001, epochs=100,
                 bufsiz=10000, batsiz=2048):
        super(AnonymG4ClassTester, self).__init__(attack_column, sensitive_column, input_veclen, sample_len,
                                                  learning_rate, 'categorical_crossentropy',
                                                  ['categorical_accuracy',
                                                   keras.metrics.AUC(name='auc', multi_label=False)],
                                                  bufsiz, epochs)
        self.BATCH_SIZE = batsiz

    def get_model(self, invec_size):
        return Goldsteen4ClassModel(keras.layers.Input(shape=(invec_size,)))

    def get_trained_model(self, x, y, fname=None):
        assert (self.input_veclen == x.shape[1])
        model = self.get_model(x.shape[1])
        model.compile(optimizer=keras.optimizers.Adam(learning_rate=self.learning_rate), loss=self.loss,
                      metrics=self.METRICS)

        resampled_ds, val_ds, resampled_steps_per_epoch = self.make_balanced_ds(x, y.to_numpy())

        model.fit(resampled_ds, epochs=self.epochs, steps_per_epoch=resampled_steps_per_epoch,
                  callbacks=[self.early_stopping()], validation_data=val_ds)

        if fname is not None:
            model.save(fname)

        return model


    def make_balanced_ds(self, x, y):
        x_train, x_valid, y_train, y_valid = train_test_split(x, y, test_size=0.2)

        labels = np.unique(y_train) # [0,1,2,3]

        ds_per_label = [self.make_ds(x_train[y_train == l], to_categorical(y_train[y_train == l], num_classes=len(labels)))
                        for l in labels]
        balanced_weights = [1.0/len(labels) for i in range(len(labels))]

        resampled_ds = tf.data.Dataset.sample_from_datasets(ds_per_label, weights=balanced_weights)
        resampled_ds = resampled_ds.batch(self.BATCH_SIZE).prefetch(2)

        min_label_num = min([(y_train == l).sum() for l in labels])

        resampled_steps_per_epoch = int(np.ceil(float(len(labels)) * min_label_num / self.BATCH_SIZE))

        val_ds = tf.data.Dataset.from_tensor_slices((x_valid, to_categorical(y_valid))).cache()
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
        pred_loss, pred_acc, _ = optmodel.evaluate(x_test_encoded, to_categorical(y_test.to_numpy()), verbose=0)
        assert (not np.isnan(pred_loss))

        prediction = optmodel.predict(x_test_encoded)
        report = classification_report(y_test, prediction.argmax(axis=1).flatten(), digits=3, output_dict=True)['macro avg']
        auc_score = roc_auc_score(to_categorical(y_test.to_numpy()), prediction)

        return [pred_acc, report['precision'], report['recall'], report['f1-score'], auc_score]

    def get_predictions(self, model, x):
        return model.predict(x).argmax(axis=1).reshape(-1,1)

    def measure_difpriv_accuracies(self, eps, categorical_features, x_train_encoded, preprocessor, y_train,
                                   x_test_encoded, y_test, model_path=None):
        dp_clf = self.get_diffpriv_classifier(eps)
        dp_clf.fit(x_train_encoded, to_categorical(y_train.to_numpy()), epochs=self.epochs)

        if model_path is not None:
            self.save_dpmodel(dp_clf, model_path)

        return self.measure_accuracies(dp_clf, x_train_encoded, y_train, x_test_encoded, y_test, preprocessor,
                                       categorical_features)

