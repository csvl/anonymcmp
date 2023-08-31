# anonymcmp tests

1. [anonymization_adult-dectree.py](anonymization_adult-dectree.py):
The same implementation as anonymization_adult-levels.ipynb.\
Anonymizations with QI:*age, education-num, race and relationship*, along differernt k and differential privacy along different epsilons for decision tree classification
1. [anonymization_adult-levels-NN1.py](anonymization_adult-levels-NN1.py): Id. for the classification based on the first NN model used in ["Anonymizing Machine Learning Models"](https://arxiv.org/abs/2007.13086)
1. [anonymization_adult-levels-NN2.py](anonymization_adult-levels-NN2.py): Id based on the second NN model in ["Anonymizing Machine Learning Models"](https://arxiv.org/abs/2007.13086)
1. [anonymization_adult-levels-NN1-Qi8.py](anonymization_adult-levels-NN1-Qi8.py): Id. with QI:*workclass, education-num, marital-status, occupation, relationship, race, sex and native-country*, for the classification based on the first NN model
1. [anonymization_adult-levels-NN1-Qi10.py](anonymization_adult-levels-NN1-Qi10.py): Id. with QI:*age, workclass, education-num, marital-status, occupation, relationship, race, sex, hours-per-week, native-country*
1. [anonymization_adult-levels-NN1-Qi12.py](anonymization_adult-levels-NN1-Qi12.py): Id. with QI:*age, workclass, education-num, marital-status, occupation, relationship, race, sex, capital-gain, capital-loss, hours-per-week, native-country*
1. [anonymization_nursery-dectree.py](anonymization_nursery-dectree.py): Anonymizations with QI:*finance, social, health*, along differernt k and differential privacy along different epsilons for decision tree classification on Nursery dataset
1. [anonymization_nursery-logreg.py](anonymization_nursery-logreg.py): Id. for Logistic Regression classification
1. [anonymization_nursery-randforest.py](anonymization_nursery-randforest.py): Id. for random forest classification
1. [anonymization_nursery-NN.py](anonymization_nursery-NN.py): Id. for the classification based on the first NN model used in ["Anonymizing Machine Learning Models"](https://arxiv.org/abs/2007.13086)
