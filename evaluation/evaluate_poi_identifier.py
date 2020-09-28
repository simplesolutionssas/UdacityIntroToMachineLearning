#!/usr/bin/python


"""
    Starter code for the evaluation mini-project.
    Start by copying your trained/tested POI identifier from
    that which you built in the validation mini-project.

    This is the second step toward building your POI identifier!

    Start by loading/formatting the data...
"""

import pickle
import sys
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import recall_score, precision_score, accuracy_score
sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit


# load data from file
file = '../final_project/final_project_dataset.pkl'
data_dict = pickle.load(open(file, 'r'))

# first element is our labels, any added elements are predictor
# features. Keep this the same for the mini-project, but you'll
# have a different feature list when you do the final project.
features_list = ["poi", "salary"]
data = featureFormat(data_dict, features_list)
labels, features = targetFeatureSplit(data)

# split data for training and testing
features_train, features_test, labels_train, labels_test = \
    train_test_split(features, labels, test_size=0.3, random_state=42)

# second classification attempt, but with training data this time.
clf = DecisionTreeClassifier()
clf.fit(features_train, labels_train)
score = clf.score(features_test, labels_test)
print('validated decision tree score: {}'.format(score))
predictions = clf.predict(features_test)

people_count = len(features_test)
true_pois = sum([label == 1 for label in labels_test])
predicted_pois = sum(predictions)
print('pois present: {}, pois predicted: {}, out of: {} people on the test set'
      .format(true_pois, predicted_pois, people_count))
print('identifier accuracy if 0 pois were predicted on test set: {:.4}'
      .format((people_count - predicted_pois)/people_count))

predictions_validation = sum([prediction == label for (prediction, label)
                             in zip(predictions, labels_test) if label == 1])
print('true positives in test set: {}'.format(predictions_validation))

# metrics
accuracy = accuracy_score(labels_test, predictions)
precision = precision_score(labels_test, predictions)
recall = recall_score(labels_test, predictions)
print('identifier accuracy: {}, precision: {}, recall: {}'
      .format(accuracy, precision, recall))

# hypothetical test set
hyp_predictions = [0, 1, 1, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 1, 0, 1, 0, 1]
true_labels = [0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 1, 0, 1, 1, 1, 0, 1, 0, 0]
true_positives = [prediction == label for (prediction, label)
                  in zip(hyp_predictions, true_labels) if label == 1]
print('true positives in test set: {}'.format(sum(true_positives)))
true_negatives = [prediction == label for (prediction, label)
                  in zip(hyp_predictions, true_labels) if label == 0]
print('true negatives in test set: {}'.format(sum(true_negatives)))
false_positives = [label == 0 for (prediction, label)
                   in zip(hyp_predictions, true_labels) if prediction == 1]
print('false positives in test set: {}'.format(sum(false_positives)))
false_negatives = [label == 1 for (prediction, label)
                   in zip(hyp_predictions, true_labels) if prediction == 0]
print('false negatives in test set: {}'.format(sum(false_negatives)))
accuracy = accuracy_score(true_labels, hyp_predictions)
precision = precision_score(true_labels, hyp_predictions)
recall = recall_score(true_labels, hyp_predictions)
print('hypothetic identifier accuracy: {}, precision: {}, recall: {}'
      .format(accuracy, precision, recall))
