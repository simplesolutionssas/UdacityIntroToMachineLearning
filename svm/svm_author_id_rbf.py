#!/usr/bin/python

"""
    This is the code to accompany the Lesson 2 (SVM) mini-project.

    Use a SVM to identify emails from the Enron corpus by their authors:
    Sara has label 0
    Chris has label 1
"""

import sys
from time import time
from sklearn.svm import SVC
sys.path.append("../tools/")
from email_preprocess import preprocess

# features_train and features_test are the features for the training
# and testing datasets, respectively
# labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()

# on this file we'll use a smaller training set (1% of the orginal)
reduced_features_train = features_train[:len(features_train)/100]
reduced_labels_train = labels_train[:len(labels_train)/100]


def train_svm(features, labels, c):
    # your code goes here
    classifier = SVC(kernel='rbf', gamma='auto', C=c)
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
    print('Model accuracy: {}, C: {}, set size: {}'.format(accuracy,
          c, len(features)))

    return predictions


# test which value of C produces the best results
c_values = [10, 100, 1000, 10000]
for c in c_values:
    predictions = train_svm(reduced_features_train, reduced_labels_train, c)

# run the classifier with the full data set and optimized C
predictions = train_svm(features_train, labels_train, 10000)

# make specific predictions for particular test data points
test_points = [10, 26, 50]
for point in test_points:
    prediction = predictions[point]
    print('Predicted value for point {} is: {}'.format(point, prediction))

# calculate how many test emails are predicted to be from Chris
chris_email_count = sum(predictions)
print('Emails predicted to be from Chris: {}'.format(chris_email_count))
