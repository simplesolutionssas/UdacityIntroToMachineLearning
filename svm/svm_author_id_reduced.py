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

# your code goes here
reduced_classifier = SVC(kernel='linear')
# fit the model and time it
start_time = time()
reduced_classifier.fit(reduced_features_train, reduced_labels_train)
print "training time:", round(time()-start_time, 3), "s"

# make predictions for the test data, and time it
start_time = time()
reduced_predictions = reduced_classifier.predict(features_test)
print "testing time:", round(time()-start_time, 3), "s"

# calculate and print the accuracy of the model
accuracy = reduced_classifier.score(features_test, labels_test)
print('Model accuracy is: {}'.format(accuracy))
