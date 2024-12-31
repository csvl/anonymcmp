from art.attacks.inference.membership_inference import MembershipInferenceBlackBox
from art.attacks.inference.attribute_inference import AttributeInferenceBlackBox
from art.attacks.inference.attribute_inference import AttributeInferenceWhiteBoxLifestyleDecisionTree
from art.attacks.inference.attribute_inference import AttributeInferenceWhiteBoxDecisionTree
from art.estimators.classification.classifier import ClassifierMixin
from sklearn.metrics import roc_auc_score
import numpy as np


# x_train, and x_test must be numerical (hot encoded for categorical)
# attack_model_type can be nn (neural network), rf (randon forest) or gb (gradient boosting)
def measure_membership_attack(classifier, x_train, y_train, x_test, y_test, attack_train_size, attack_test_size, attack_model_type='rf'):

    assert (issubclass(classifier.__class__, ClassifierMixin))

    attack = MembershipInferenceBlackBox(classifier, attack_model_type=attack_model_type)

    attack.fit(x_train[:attack_train_size], y_train[:attack_train_size],
               x_test[:attack_test_size], y_test[:attack_test_size])

    inferred_train = attack.infer(x_train[attack_train_size:2*attack_train_size],
                                  y_train[attack_train_size:2*attack_train_size], probabilities=True)
    inferred_test = attack.infer(x_test[attack_test_size:2*attack_test_size],
                                 y_test[attack_test_size:2*attack_test_size], probabilities=True)

    bin_inferred_train = inferred_train > 0.5
    bin_inferred_test  = inferred_test > 0.5

    mmb_train_acc = np.sum(bin_inferred_train) / len(inferred_train)
    mmb_test_acc = 1 - (np.sum(bin_inferred_test) / len(inferred_test))

    accuracy = (mmb_train_acc * len(inferred_train) + mmb_test_acc * len(inferred_test)) / (len(inferred_train) + len(inferred_test))

    precision = np.sum(bin_inferred_train) / (np.sum(bin_inferred_train) + np.sum(bin_inferred_test))
    recall = mmb_train_acc
    f1 = 2 * precision * recall / (precision + recall)

    act_values = np.concatenate((np.ones(len(inferred_train)), np.zeros(len(inferred_test))))
    inferred = np.concatenate((inferred_train, inferred_test))
    auc_score = roc_auc_score(act_values, inferred)

    return accuracy, precision, recall, f1, auc_score


# x_train  must be numerical (hot encoded for categorical)
def measure_attribute_bbox_attack_accuracy(classifier, x_train, attack_train_size, predictions, attack_feature,
                                           x_train_for_attack, values, x_train_feature):

    assert (issubclass(classifier.__class__, ClassifierMixin))

    attack = AttributeInferenceBlackBox(classifier, attack_feature=attack_feature)

    attack.fit(x_train[:attack_train_size])

    inferred_train = attack.infer(x_train_for_attack[attack_train_size:], pred=predictions[attack_train_size:],
                                  values=values)

    return np.sum(np.around(inferred_train, decimals=4) == np.around(x_train_feature[attack_train_size:], decimals=4).reshape(1, -1)) / len(inferred_train)


def measure_attribute_wboxLDT_attack_accuracy(classifier, predictions, attack_feature, x_train_for_attack, values,
                                             priors, x_train_feature):

    assert (issubclass(classifier.__class__, ClassifierMixin))

    attack = AttributeInferenceWhiteBoxLifestyleDecisionTree(classifier, attack_feature=attack_feature)

    inferred_train = attack.infer(x_train_for_attack, predictions, values=values, priors=priors)

    return np.sum(np.around(inferred_train, decimals=4) == np.around(x_train_feature, decimals=4).reshape(1, -1)) / len(inferred_train)


def measure_attribute_wboxDT_attack_accuracy(classifier, predictions, attack_feature, x_train_for_attack, values,
                                             priors, x_train_feature):

    assert (issubclass(classifier.__class__, ClassifierMixin))
    attack = AttributeInferenceWhiteBoxDecisionTree(classifier, attack_feature=attack_feature)

    inferred_train = attack.infer(x_train_for_attack, predictions, values=values, priors=priors)

    return np.sum(np.around(inferred_train, decimals=4) == np.around(x_train_feature, decimals=4).reshape(1, -1)) / len(inferred_train)
