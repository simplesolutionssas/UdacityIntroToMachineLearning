#!/usr/bin/python


"""
    Starter code for the validation mini-project.
    The first step toward building your POI identifier!

    Start by loading/formatting the data

    After that, it's not our code anymore--it's yours!
"""

import pickle
import sys
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
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

# first classification attempt, without testing data.
clf = DecisionTreeClassifier()
clf.fit(features, labels)
score = clf.score(features, labels)
print('initial decision tree score: {}'.format(score))

# split data for training and testing
features_train, features_test, labels_train, labels_test = \
    train_test_split(features, labels, test_size=0.3, random_state=42)

# second classification attempt, but with training data this time.
improved_clf = DecisionTreeClassifier()
improved_clf.fit(features_train, labels_train)
score = improved_clf.score(features_test, labels_test)
print('validated decision tree score: {}'.format(score))
