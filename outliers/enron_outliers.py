#!/usr/bin/python

import pickle
import sys
import matplotlib.pyplot
from pandas import DataFrame
import numpy as np
sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit


# read in data dictionary, convert to numpy array
file_path = '../final_project/final_project_dataset.pkl'
file = open(file_path, 'r')
enron_data = pickle.load(file)
file.close()
features = ['salary', 'bonus']

# identify and remove the outliers that don't belong to the dataset
enron_df = DataFrame.from_dict(enron_data, orient='index')
enron_df.replace('NaN', np.nan, regex=True, inplace=True)
enron_df.sort_values('salary', ascending=False).head()
enron_data.pop('TOTAL', 0)

# format the features for plot use
data = featureFormat(enron_data, features)

# visualize the data
for point in data:
    salary = point[0]
    bonus = point[1]
    matplotlib.pyplot.scatter(salary, bonus)

matplotlib.pyplot.xlabel('salary')
matplotlib.pyplot.ylabel('bonus')
matplotlib.pyplot.show()
