from art.attacks.inference.membership_inference import MembershipInferenceBlackBox
from art.attacks.inference.attribute_inference import AttributeInferenceBlackBox
from art.attacks.inference.attribute_inference import AttributeInferenceWhiteBoxLifestyleDecisionTree
from art.attacks.inference.attribute_inference import AttributeInferenceWhiteBoxDecisionTree
from art.estimators.classification.classifier import ClassifierMixin
import numpy as np


# x_train, and x_test must be numerical (hot encoded for categorical)
# attack_model_type can be nn (neural network), rf (randon forest) or gb (gradient boosting)
def measure_membership_attack_accuracy(classifier, x_train, y_train, x_test, y_test, attack_train_size,
                                       attack_test_size, attack_model_type='rf'):

    assert (issubclass(classifier.__class__, ClassifierMixin))

    attack = MembershipInferenceBlackBox(classifier, attack_model_type=attack_model_type)

    attack.fit(x_train[:attack_train_size], y_train[:attack_train_size],
               x_test[:attack_test_size], y_test[:attack_test_size])

    inferred_train = attack.infer(x_train[attack_train_size:], y_train[attack_train_size:])
    inferred_test = attack.infer(x_test[attack_test_size:], y_test[attack_test_size:])

    mmb_train_acc = np.sum(inferred_train) / len(inferred_train)
    mmb_test_acc = 1 - (np.sum(inferred_test) / len(inferred_test))

    return (mmb_train_acc * len(inferred_train) + mmb_test_acc * len(inferred_test)) / (len(inferred_train) + len(inferred_test))


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
