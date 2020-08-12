#!/usr/bin/python

"""
    Starter code for the regression mini-project

    Loads up/formats a modified version of the dataset
    (why modified?  we've removed some trouble points
    that you'll find yourself in the outliers mini-project).

    Draws a little scatterplot of the training/testing data

    You fill in the regression code where indicated:
"""

import pickle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import sys
sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit

# load the final_project_dataset_modified.pkl file
file_path = '../final_project/final_project_dataset_modified.pkl'
file = open(file_path, 'r')
dictionary = pickle.load(file)
file.close()

# list the features you want to look at
# first item in the list will be the "target" feature
features_list = ['bonus', 'salary']
# you can switch the previous line with the next one to use another feature to
# try and predict people's bonuses
# features_list = ['bonus', 'long_term_incentive']
data = featureFormat(dictionary, features_list, remove_any_zeroes=True)
target, features = targetFeatureSplit(data)

# training-testing split needed in regression, just like classification
feature_train, feature_test, target_train, target_test = \
            train_test_split(features, target, test_size=0.5, random_state=42)
train_color = 'b'
test_color = 'r'

# regression code. named reg, so that plotting code below plots it correctly
reg = LinearRegression()
reg.fit(feature_train, target_train)
prediction_test = reg.predict(feature_test)
print('train fit - slope: {}, intercept: {}'.format(reg.coef_, reg.intercept_))
train_score = reg.score(feature_train, target_train)
test_score = reg.score(feature_test, target_test)
print('train score: {:.2%}, test score: {:.2%}'.format(train_score, test_score))

# draw the scatterplot, with color-coded training and testing points
for feature, target in zip(feature_test, target_test):
    plt.scatter(feature, target, color=test_color)
for feature, target in zip(feature_train, target_train):
    plt.scatter(feature, target, color=train_color)

# labels for the legend
plt.scatter(feature_test[0], target_test[0], color=test_color, label='test')
plt.scatter(feature_test[0], target_test[0], color=train_color, label='train')

# draw the regression line, once it's coded
try:
    plt.plot(feature_test, prediction_test)
except NameError:
    pass

# to see the effect of outliers, we now fit and plot using the test data to
# compare to the fit and plot of the train data. you can see that the biggest
# outliers on the plot are blue, which means that the training set includes
# them while fitting the data which translates to a worse score
reg.fit(feature_test, target_test)
prediction_test = reg.predict(feature_test)
print('test fit - slope: {}, intercept: {}'.format(reg.coef_, reg.intercept_))
train_score = reg.score(feature_train, target_train)
test_score = reg.score(feature_test, target_test)
print('train score: {:.2%}, test score: {:.2%}'.format(train_score, test_score))
plt.plot(feature_train, reg.predict(feature_train), color='b')

plt.xlabel(features_list[1])
plt.ylabel(features_list[0])
plt.legend()
plt.show()
