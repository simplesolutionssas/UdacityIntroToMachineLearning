#!/usr/bin/python

import random
import numpy as np
import matplotlib.pyplot as plt
import pickle
from outlier_cleaner import outlierCleaner
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# load up some practice data with outliers in it
outliers_ages_file_path = '../outliers/practice_outliers_ages.pkl'
outliers_ages_file = open(outliers_ages_file_path, 'r')
ages = pickle.load(outliers_ages_file)
outliers_ages_file.close()
net_worths_file_path = '../outliers/practice_outliers_net_worths.pkl'
net_worths_file = open(net_worths_file_path, 'r')
net_worths = pickle.load(net_worths_file)
net_worths_file.close()

# ages and net_worths need to be reshaped into 2D numpy arrays. second argument
# of reshape command is a tuple of integers: (n_rows, n_columns). by convention
# n_rows is the number of data points and n_columns is the number of features
ages = np.reshape(np.array(ages), (len(ages), 1))
net_worths = np.reshape(np.array(net_worths), (len(net_worths), 1))
ages_train, ages_test, net_worths_train, net_worths_test = \
            train_test_split(ages, net_worths, test_size=0.1, random_state=42)

# fill in a regression here!  Name the regression object reg so that the
# plotting code below works, and you can see what your regression looks like
reg = LinearRegression()
reg.fit(ages_train, net_worths_train)
print('train fit - slope: {}, intercept: {}'.format(reg.coef_, reg.intercept_))
train_score = reg.score(ages_test, net_worths_test)
print('train score: {:.2%}'.format(train_score))

try:
    plt.plot(ages, reg.predict(ages), color='blue')
except NameError:
    pass
plt.scatter(ages, net_worths)
plt.show()

# identify and remove the most outlier-y points
cleaned_data = []
try:
    predictions = reg.predict(ages_train)
    cleaned_data = outlierCleaner(predictions, ages_train, net_worths_train)
except NameError:
    print("your regression object doesn't exist, or isn't name reg")
    print("can't make predictions to use in identifying outliers")

# only run this code if cleaned_data is returning data
if len(cleaned_data) > 0:
    ages, net_worths, errors = zip(*cleaned_data)
    ages = np.reshape(np.array(ages), (len(ages), 1))
    net_worths = np.reshape(np.array(net_worths), (len(net_worths), 1))

    # refit your cleaned data!
    try:
        reg.fit(ages, net_worths)
        print('new train fit - slope: {}, intercept: {}'.
              format(reg.coef_, reg.intercept_))
        train_score = reg.score(ages_test, net_worths_test)
        print('new train score: {:.2%}'.format(train_score))
        plt.plot(ages, reg.predict(ages), color='blue')
    except NameError:
        print ("you don't seem to have regression imported/created,")
        print ("or else your regression object isn't named reg")
        print ("either way, only draw the scatter plot of the cleaned data")
    plt.scatter(ages, net_worths)
    plt.xlabel('ages')
    plt.ylabel('net worths')
    plt.show()

else:
    print("outlierCleaner() is returning an empty list, no refitting was done")
