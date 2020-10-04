#!/usr/bin/python

import sys
import numpy as np
from numpy import core
import pandas as pd
import pickle
from matplotlib import pyplot as plt
from pandas import DataFrame
from sklearn.metrics.classification import f1_score, precision_score
from sklearn.metrics.classification import recall_score
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from tester import dump_classifier_and_data
sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit


# Load the dictionary containing the dataset
file_path = 'final_project_dataset.pkl'
file = open(file_path, 'r')
enron_data = pickle.load(file)
file.close()

# Remove the TOTAL row from the dataset
enron_data.pop('TOTAL', 0)
print('Enron data point count: {}'.format(len(enron_data)))

# Put the dictionary in a DataFrame and perform some cleaning operations
pd.options.display.float_format = '{:20,.2f}'.format
enron_df = DataFrame.from_dict(enron_data, orient='index')
# All NaN strings are converted to Numpy nan values, which allows the describe
# function to produce proper numeric values for all statistics. 
enron_df.replace('NaN', np.nan, regex=True, inplace=True)
# Convert True to 1 and False to 0.
enron_df.replace({True: 1, False: 0}, inplace=True)
enron_df.drop('email_address', axis=1, inplace=True)
enron_df.head()
enron_df.describe()

# Task 1: Select what features you'll use.
# financial features (all units are in US dollars):
#   'salary', 'deferral_payments', 'total_payments', 'loan_advances', 'bonus',
#   'restricted_stock_deferred', 'deferred_income', 'total_stock_value',
#   'expenses', 'exercised_stock_options', 'other', 'long_term_incentive',
#   'restricted_stock', 'director_fees'
# email features (‘email_address’ is string, the rest, email message counts):
#   'email_address', 'to_messages', 'from_poi_to_this_person', 'from_messages',
#   'from_this_person_to_poi', 'shared_receipt_with_poi']
# POI label: (boolean, represented as integer)
#   ‘poi’
# features_list is a list of strings, each of which is a feature name.
# The first feature must be "poi". You will need to use more features.
features_list = ['poi', 'salary', 'deferral_payments', 'total_payments',
                 'loan_advances', 'bonus', 'restricted_stock_deferred',
                 'deferred_income', 'total_stock_value', 'expenses',
                 'exercised_stock_options', 'other', 'long_term_incentive',
                 'restricted_stock', 'director_fees', 'to_messages',
                 'from_poi_to_this_person', 'from_messages',
                 'from_this_person_to_poi', 'shared_receipt_with_poi']

# Plot the variables to understand them better.
# for col in enron_df.columns:
#     enron_df.hist(column=col, bins=100, alpha=0.5)


# Task 2: Remove outliers

# Task 3: Create new feature(s)
# Store to my_dataset for easy export below.
my_dataset = enron_data

# Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys=True)
labels, features = targetFeatureSplit(data)
labels = np.array(labels)
features = np.array(features)

# Task 4: Try a variety of classifiers
# Please name your classifier clf for easy export below.
# Note that if you want to do PCA or other multi-stage operations,
# you'll need to use Pipelines. For more info:
# http://scikit-learn.org/stable/modules/pipeline.html


# Task 5: Tune your classifier to achieve better than .3 precision and recall
# using our testing script. Check the tester.py script in the final project
# folder for details on the evaluation method, especially the test_classifier
# function. Because of the small size of the dataset, the script uses
# stratified shuffle split cross validation. For more info:
# http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html




# Pipeline:
# 1. scale
# 2. reduce_dim
# 3. stratified_shuffle_split
# 4. classify


# for train_index, test_index in sss.split(features, labels):
#     features_train, features_test = features[train_index], features[test_index]
#     labels_train, labels_test = labels[train_index], labels[test_index]

# features_train, features_test, labels_train, labels_test = \
#     train_test_split(features, labels, test_size=0.3, random_state=42)

# all stages are populated by the param_grid
pipe = Pipeline([('reduce_dim', PCA()),
                 ('classify', DecisionTreeClassifier(random_state=42))])

pca_n_components = [2, 4, 8, 16]

param_grid = [
    {
        'reduce_dim': [PCA()],
        'reduce_dim__n_components': pca_n_components,
    }
]

# For cross-validation we'll use a stratified shuffle split because of the
# small size of the dataset.
cv_strategy = StratifiedShuffleSplit(n_splits=1000, random_state=42)

# We define all the scoring metrics we want to measure. But Recall will be the
# one used to select the best set of parameters, and refit the identifier at
# the end, because in my opinion false positives are far better than false
# negatives in this case, since we don't want to produce any false negatives
# and risk missing a poi.
scoring = {'Accuracy': 'accuracy', 'Recall': 'recall',
           'Precision': 'precision', 'F1 Score': 'f1'}

clf = GridSearchCV(pipe, n_jobs=8, param_grid=param_grid, cv=cv_strategy,
                   scoring=scoring, iid=False, refit='Recall', verbose=2,
                   return_train_score=True)
best_model = clf.fit(features, labels)
results = clf.cv_results_
print('\nBest estimator:\n{}\n'.format(clf.best_estimator_))
for metric_name, metric in scoring.items():
    result_name = 'mean_test_' + metric_name
    result = results[result_name]
    print('Best estimator {}: {}'.format(metric, result))

plt.title("GridSearchCV results",
          fontsize=16)

plt.xlabel("reduce_dim__n_components")
plt.ylabel("Score")

ax = plt.gca()
ax.set_xlim(0, 16)
ax.set_ylim(0.0, 1)

# Get the regular numpy array from the MaskedArray
X_axis = np.array(results['param_reduce_dim__n_components'].data, dtype=float)

for scorer, color in zip(sorted(scoring), ['g', 'k', 'b', 'r']):
    for sample, style in (('train', '--'), ('test', '-')):
        sample_score_mean = results['mean_%s_%s' % (sample, scorer)]
        sample_score_std = results['std_%s_%s' % (sample, scorer)]
        ax.fill_between(X_axis, sample_score_mean - sample_score_std,
                        sample_score_mean + sample_score_std,
                        alpha=0.1 if sample == 'test' else 0, color=color)
        ax.plot(X_axis, sample_score_mean, style, color=color,
                alpha=1 if sample == 'test' else 0.7,
                label="%s (%s)" % (scorer, sample))

    best_index = np.nonzero(results['rank_test_%s' % scorer] == 1)[0][0]
    best_score = results['mean_test_%s' % scorer][best_index]

    # Plot a dotted vertical line at the best score for that scorer marked by x
    ax.plot([X_axis[best_index], ] * 2, [0, best_score],
            linestyle='-.', color=color, marker='x', markeredgewidth=3, ms=8)

    # Annotate the best score for that scorer
    ax.annotate("%0.2f" % best_score,
                (X_axis[best_index], best_score + 0.005))

plt.legend(loc="best")
plt.grid(False)
plt.show()


experiment_definitions = {
    'sklearn.naive_bayes.GaussianNB':
        {},
    'sklearn.tree.DecisionTreeClassifier':
        {
            'criterion': ['entropy', 'gini'],
            'splitter': ['best', 'random'],
            'min_samples_split': [2, 4, 8, 16, 32, 64]
        },
    'sklearn.svm.SVC':
        {
            'kernel': ['linear', 'rbf', 'sigmoid', 'poly'],
            'gamma': ['auto', 'scale'],
            'C': [10, 100, 1000, 10000],
            'degree': [2, 3, 4, 5]
         },
    'sklearn.neighbors.KNeighborsClassifier':
        {
            'n_neighbors': [2, 4, 8, 16, 32, 64],
            'weights': ['uniform', 'distance'],
            'algorithm': ['ball_tree', 'kd_tree', 'brute'],
            'p': [1, 2]
        },
    'sklearn.ensemble.RandomForestClassifier':
        {
            'n_estimators': [8, 16, 32, 64],
            'criterion': ['entropy', 'gini'],
            'min_samples_split': [2, 4, 8, 16, 32, 64],
            'max_depth': [2, 4, 8, 16],
            'max_features': [None, 'sqrt', 'log2']
        },
    'sklearn.ensemble.AdaBoostClassifier':
        {
            'base_estimator': [None,
                               SVC(kernel='poly', gamma='scale', degree=5),
                               DecisionTreeClassifier(splitter='random')],
            'n_estimators': [8, 16, 32, 64, 128],
            'algorithm': ['SAMME'],
            'learning_rate': [0.01, 0.05, 0.1, 0.3, 1]
        },
    'sklearn.cluster.KMeans':
        {
            'n_clusters': [2, 4, 6, 8, 16]
        }
}

results = DataFrame.from_dict(clf.cv_results_)
results.head()

# Task 6: Dump your classifier, dataset, and features_list so anyone can check
# your results. You do not need to change anything below, but make sure that
# the version of poi_id.py that you submit can be run on its own and generates
# the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, features_list)
