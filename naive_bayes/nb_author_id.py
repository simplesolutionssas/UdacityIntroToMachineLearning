#!/usr/bin/python

import sys
from time import time
from sklearn.naive_bayes import GaussianNB
sys.path.append("../tools/")
from email_preprocess import preprocess

"""
    This is the code to accompany the Lesson 1 (Naive Bayes) mini-project.

    Use a Naive Bayes Classifier to identify emails by their authors

    authors and labels:
    Sara has label 0
    Chris has label 1
"""

# features_train and features_test are the features for the training
# and testing datasets, respectively
# labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()

# create, train and use the classifier to make predictions on
classifier = GaussianNB()
# fit the model and time it
start_time = time()
classifier.fit(features_train, labels_train)
print "training time:", round(time()-start_time, 3), "s"

# make predictions for the test data, and time it
start_time = time()
predictions = classifier.predict(features_test)
print "testing time:", round(time()-start_time, 3), "s"

# calculate and print the accuracy of the model
accuracy = classifier.score(features_test, labels_test)
print('Model accuracy is: {}'.format(accuracy))
