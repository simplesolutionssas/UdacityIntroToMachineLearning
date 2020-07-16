#!/usr/bin/python

"""
    This is the code to accompany the Lesson 3 (decision tree) mini-project.

    Use a Decision Tree to identify emails from the Enron corpus by author:    
    Sara has label 0
    Chris has label 1
"""

import sys
from time import time
from sklearn.tree import DecisionTreeClassifier
sys.path.append("../tools/")
from email_preprocess import preprocess

# features_train and features_test are the features for the training
# and testing datasets, respectively
# labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()
print('number of features: {}'.format(len(features_train[0])))


def train_tree(features, labels, criterion, min_samples_split):
    # your code goes here
    classifier = DecisionTreeClassifier(criterion=criterion,
                                        min_samples_split=min_samples_split)
    # fit the model and time it
    start_time = time()
    classifier.fit(features, labels)
    print("\ntraining time: {} s".format(round(time()-start_time, 3)))

    # measure the prediction time for the test data
    start_time = time()
    predictions = classifier.predict(features_test)
    print("testing time: {} s".format(round(time()-start_time, 3)))

    # calculate and print the accuracy of the model
    accuracy = classifier.score(features_test, labels_test)
    print('model accuracy: {}, criterion: {}, sample split: {}, set size: {}'
          .format(accuracy, criterion, min_samples_split, len(features)))

    return predictions


# test which value of C produces the best results
min_sample_splits = [2, 20, 40, 80]
criteria = ['entropy', 'gini']
for split in min_sample_splits:
    for criterion in criteria:
        predictions = train_tree(features_train, labels_train,
                                 criterion, split)

# make specific predictions for particular test data points
test_points = [10, 26, 50]
for point in test_points:
    prediction = predictions[point]
    print('\npredicted value for point {} is: {}'.format(point, prediction))

# calculate how many test emails are predicted to be from Chris
chris_email_count = sum(predictions)
print('\nemails predicted to be from Chris: {}'.format(chris_email_count))
