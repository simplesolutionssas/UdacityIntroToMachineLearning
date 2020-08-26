#!/usr/bin/python

"""
    Skeleton code for k-means clustering mini-project.
"""

import pickle
import numpy as np
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import sys
from sklearn.cluster import KMeans
sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit


def Draw(pred, features, poi, mark_poi=False, name="image.png",
         f1_name="feature 1", f2_name="feature 2"):
    """ some plotting code designed to help you visualize your clusters """

    # plot each cluster with a different color--add more colors for
    # drawing more than five clusters
    colors = ["b", "c", "k", "m", "g"]
    for ii, pp in enumerate(pred):
        plt.scatter(features[ii][0], features[ii][1], color=colors[pred[ii]])

    # if you like, place red stars over points that are POIs (just for funsies)
    if mark_poi:
        for ii, pp in enumerate(pred):
            if poi[ii]:
                plt.scatter(features[ii][0], features[ii][1], color="r",
                            marker="*")
    plt.xlabel(f1_name)
    plt.ylabel(f2_name)
    plt.savefig(name)
    plt.show()


# load the dict of dicts containing all the data on each person in the dataset
file_path = '../final_project/final_project_dataset.pkl'
file = open(file_path, 'r')
enron_data = pickle.load(file)
file.close()
# there's an outlier--remove it!
enron_data.pop("TOTAL", 0)

# the input features can be any key in the person-level dictionary
# (salary, director_fees, etc.)
feature_1 = "salary"
feature_2 = "exercised_stock_options"
feature_3 = "total_payments"
poi = "poi"
features_list = [poi, feature_1, feature_2, feature_3]
data = featureFormat(enron_data, features_list)
poi, finance_features = targetFeatureSplit(data)

# in the "clustering with 3 features" part of the mini-project, you'll want to
# change this line to for f1, f2, _ in finance_features:
# (as it's currently written, the line below assumes 2 features)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
for f1, f2, f3 in finance_features:
    ax.scatter(f1, f2, f3, c='r', marker='o')
plt.show()

# create predictions of the cluster labels for the data and store them on pred
k_means = KMeans(n_clusters=2)
pred = k_means.fit_predict(finance_features)

# rename the "name" parameter when you change the number of features
# so that the figure gets saved to a different file
try:
    Draw(pred, finance_features, poi, mark_poi=False, name="clusters.pdf",
         f1_name=feature_1, f2_name=feature_2)
except NameError:
    print('no predictions object named pred found, no clusters to plot')

# construct and explore the enron_df DataFrame
enron_df = pd.DataFrame.from_dict(enron_data, orient='index')
# convert the 'Nan' strings into real NaN values
enron_df.replace('NaN', np.nan, regex=True, inplace=True)
# describe the exercised_stock_options DataFrame series
print(enron_df.exercised_stock_options.describe())
# describe the salary DataFrame series
print(enron_df.salary.describe())
