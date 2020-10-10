#!/usr/bin/python

import sys
import numpy as np
import pickle
import pandas as pd
import seaborn as sns
from pandas import DataFrame
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.cluster import KMeans
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedShuffleSplit
from tester import dump_classifier_and_data
sys.path.append('../tools/')
from feature_format import featureFormat, targetFeatureSplit


def load_data(file_path):
    '''
    Retrieve the dataset stored in the specific pickle file path.

    Args:
        file_path : string
            The absolute file path for the pickle file containing the data.

    Returns:
        dataset : dictionary
            Dictionary containing the data stored in the file, in a structured
            format.
    '''
    # Load the dictionary containing the dataset
    file = open(file_path, 'r')
    dataset = pickle.load(file)
    file.close()
    return dataset


def get_enron_feature_list():
    '''
    Retrieve the feature list to be used for the Enron POI classification
    problem:

    Financial features (all units are in US dollars):
        salary, deferral_payments, total_payments, loan_advances, bonus,
        restricted_stock_deferred, deferred_income, total_stock_value,
        expenses, exercised_stock_options, other, long_term_incentive,
        restricted_stock, director_fees
    Email features ('email_address' is string, the rest, email message counts):
        email_address, to_messages, from_poi_to_this_person, from_messages,
        from_this_person_to_poi, shared_receipt_with_poi
    POI label (boolean, represented as integer):
        poi

    Args:
        None

    Returns:
        features_list : list
            The list of features that will be used for solving the POI
            classification problem.
    '''
    # The first feature must be 'poi'.
    features_list = ['poi', 'salary', 'deferral_payments', 'total_payments',
                     'loan_advances', 'bonus', 'restricted_stock_deferred',
                     'deferred_income', 'total_stock_value', 'expenses',
                     'exercised_stock_options', 'other', 'long_term_incentive',
                     'restricted_stock', 'director_fees', 'to_messages',
                     'from_poi_to_this_person', 'from_messages',
                     'from_this_person_to_poi', 'shared_receipt_with_poi']
    return features_list


def get_enron_labels_features(enron_data, enron_feature_list):
    '''
    Retrieve the labels and features for the Enron dataset, after applying
    some cleaning operations.

    Args:
        enron_data : dictionary
            Dictionary containing the data stored in the file, in a structured
            format.
        enron_feature_list : list
            The list of features that needs to be extracted from the dictionary
            and returned for the classification problem. The first feature on
            the list needs to contain the data labels. 

    Returns:
        labels : ndarray
            Array with the labels for each data point in the enron dataset.
        features : ndarray
            Array with the features for each data point in the enron dataset.
    '''
    # Remove the TOTAL row from the dataset
    enron_data.pop('TOTAL', 0)
    data = featureFormat(enron_data, enron_feature_list, sort_keys=True)
    labels, features = targetFeatureSplit(data)
    labels = np.array(labels)
    features = np.array(features)
    print('Enron Data Point Count: {}'.format(len(enron_data)))
    return labels, features


def remove_enron_outliers(labels, features):
    '''
    Return the labels and features for the Enron dataset, after eliminating the
    outlier data points from the different features.

    Args:
        labels : ndarray
            Array with the labels for each data point in the enron dataset.
        features : ndarray
            Array with the features for each data point in the enron dataset.

    Returns:
        labels : ndarray
            Array with the labels for each data point, after having removed
            all outliers from them.
        features : ndarray
            Array with the features for each data point, after having removed
            all outliers from them.

    '''
    # TODO Task 2: Remove outliers
    return labels, features


def add_enron_features(labels, features):
    '''
    Return the labels and features for the Enron dataset, after adding new
    relevant features to help improve the classification performance.

    Args:
        labels : ndarray
            Array with the labels for each data point in the enron dataset.
        features : ndarray
            Array with the features for each data point in the enron dataset.

    Returns:
        labels : ndarray
            Array with the labels for each data point, after adding the new
            features.
        features : ndarray
            Array with the features for each data point, after adding the new
            features.
    '''
    # TODO Task 3: Create new feature(s)
    return labels, features


def get_pipelines():
    '''
    Build the different pipelines that will be used to train and finetune the
    classification model.

    Args:
        None

    Returns:
        pipelines : dictionary
            A dictionary containing all the pipelines that will be used to fit
            the model in order to select the one that produces the best results
            for the given problem.
    '''
    # Pipeline:
    # 1. scale
    # 2. reduce_dim
    # 3. stratified_shuffle_split
    # 4. classify

    pipelines = {
        'GaussianNB': {
            'pipe': [('reduce_dim', PCA()),
                     ('classify', GaussianNB())],
            'param_grid': [{
                'reduce_dim': [PCA()],
                'reduce_dim__n_components': [2, 4, 8, 16],
            }]
        },
        'DecisionTreeClassifier': {
            'pipe': [('reduce_dim', PCA()),
                     ('classify', DecisionTreeClassifier(random_state=42))],
            'param_grid': [{
                'reduce_dim': [PCA()],
                'reduce_dim__n_components': [2, 4, 8, 16],
                'classify__criterion': ['entropy', 'gini'],
                'classify__splitter': ['best', 'random'],
                'classify__min_samples_split': [2, 4, 8, 16, 32, 64]
            }]
        },
        'SVC': {
            'pipe': [('reduce_dim', PCA()),
                     ('classify', SVC(random_state=42))],
            'param_grid': [{
                # I wasn't able to make SVC work with the 'linear' kernel.
                # 'reduce_dim': [PCA()],
                # 'reduce_dim__n_components': [2, 4, 8, 16],
                # 'classify__kernel': ['linear'],
                # 'classify__C': [10, 100, 1000, 10000]
            # }, {
                'reduce_dim': [PCA()],
                'reduce_dim__n_components': [2, 4, 8, 16],
                'classify__kernel': ['rbf'],
                'classify__gamma': ['auto', 'scale'],
                'classify__C': [10, 100, 1000, 10000],
            }, {
                'reduce_dim': [PCA()],
                'reduce_dim__n_components': [2, 4, 8, 16],
                'classify__kernel': ['sigmoid'],
                'classify__gamma': ['auto', 'scale'],
                'classify__C': [10, 100, 1000, 10000]
            }, {
                'reduce_dim': [PCA()],
                # With values greater than, 6 (8, 16) the search won't finish.
                'reduce_dim__n_components': [2, 4, 6],
                'classify__kernel': ['poly'],
                'classify__gamma': ['auto', 'scale'],
                'classify__C': [10, 100, 1000, 10000],
                # With a value of 2 the search won't finish.
                'classify__degree': [3, 4, 5]
            }]
        },
        'KNeighborsClassifier': {
            'pipe': [('reduce_dim', PCA()),
                     ('classify', KNeighborsClassifier())],
            'param_grid': [{
                'reduce_dim': [PCA()],
                'reduce_dim__n_components': [2, 4, 8, 16],
                'classify__n_neighbors': [2, 4, 8, 16, 32, 64],
                'classify__weights': ['uniform', 'distance'],
                'classify__algorithm': ['ball_tree', 'kd_tree', 'brute'],
                'classify__p': [1, 2]
            }]
        },
        'RandomForestClassifier': {
            'pipe': [('reduce_dim', PCA()),
                     ('classify', RandomForestClassifier(random_state=42))],
            'param_grid': [{
                'reduce_dim': [PCA()],
                'reduce_dim__n_components': [2, 4, 8, 16],
                'classify__n_estimators': [8, 16, 32, 64],
                'classify__criterion': ['entropy', 'gini'],
                'classify__min_samples_split': [2, 4, 8, 16, 32, 64],
                'classify__max_depth': [2, 4, 8, 16],
                'classify__max_features': [None, 'sqrt', 'log2']
            }]
        },
        'AdaBoostClassifier': {
            'pipe': [('reduce_dim', PCA()),
                     ('classify', AdaBoostClassifier(random_state=42))],
            'param_grid': [{
                'reduce_dim': [PCA()],
                'reduce_dim__n_components': [2, 4, 8, 16],
                'classify__base_estimator': [
                    None,
                    SVC(kernel='poly', gamma='scale', degree=5),
                    DecisionTreeClassifier(splitter='random')
                ],
                'classify__n_estimators': [8, 16, 32, 64, 128],
                'classify__algorithm': ['SAMME'],
                'classify__learning_rate': [0.01, 0.05, 0.1, 0.3, 1]
            }],
        },
        'KMeans': {
            'pipe': [('reduce_dim', PCA()),
                     ('classify', KMeans(random_state=42))],
            'param_grid': [{
                'reduce_dim': [PCA()],
                'reduce_dim__n_components': [2, 4, 8, 16],
                'classify__n_clusters': [2]
            }]
        }
    }

    return pipelines


def get_best_estimator_metrics(results, metrics):
    '''
    Process the search results DataFrame and extract from it the metrics for
    the best estimator.

    Args:
        results : DataFrame
            DataFrame with the results of the grid search.
        metrics : list
            List containing the names of the metrics evaluated for the
            estimator during the search. The first metric in the list is
            assumed to be the main metric, which was used to select the best
            estimator.

    Returns:
        estimator_metrics : list
            List containing the best estimator's values for the metrics
            evaluated during the search.
    '''
    estimator_metrics = []
    best_estimator_string = 'Best Estimator {}: {:.4f}'

    main_metric_name = 'mean_test_' + metrics[0]
    main_metric_results = results[main_metric_name]
    main_metric_value = max(main_metric_results)
    main_metric_index = np.argmax(main_metric_results, axis=0)
    print(best_estimator_string.format(metrics[0].title(), main_metric_value))
    estimator_metrics.append(main_metric_value)

    for metric in metrics[1:]:
        full_metric_name = 'mean_test_' + metric
        metric_results = results[full_metric_name]
        metric_value = metric_results[main_metric_index]
        print(best_estimator_string.format(metric.title(), metric_value))
        estimator_metrics.append(metric_value)

    return estimator_metrics


def add_best_metric_value_marker(results, axe, x_values, metric, color):
    '''
    For a metric, plot a dotted vertical line marked with an x at the best
    score obtained, and annotate it with the value for that score.

    Args:
        results : DataFrame
            DataFrame with the results of the grid search.
        axe : Axes
            Axe where we'll plot the dotted vertical line.
        x_values : ndarray
            Array with the values used for the chart's X axis.
        metric : string
            The name of the metric whose best value we want to mark.
        color : string
            The code of the color we want to mark the best value with.

    Returns:
        None
    '''
    best_index = np.nonzero(results['rank_test_%s' % metric] == 1)[0][0]
    best_score = results['mean_test_%s' % metric][best_index]
    axe.plot([x_values[best_index], ] * 2, [0, best_score], linestyle='-.',
             color=color, marker='x', markeredgewidth=3, ms=8)
    axe.annotate('%0.2f' % best_score,
                 (x_values[best_index], best_score + 0.005))


def plot_estimator_metrics(estimator, metrics, results):
    '''
    Generate a graphic graphic comparing the results obtained for each one of
    the different candidates, for each one of the different scoring metrics
    used for the estimator search.

    Args:
        estimator : string
            The name of the estimator whose results are going to be plotted.
        results : DataFrame
            DataFrame with the results of the grid search.

    Returns:
        None
    '''
    # TODO explore with a pivot table in pandas that gets the average metric
    # score (for all metrics evaluated) for each value of a parameter. This
    # could be then plotted to see the impact of the specific parameter on
    # the results. Iterating over the different parameters we would end up
    # with a group of charts (one per parameter) to detect those parameters
    # most important for solving the particular problem.

    # TODO: generalize this code to accept more than 16 experiments.

    main_metric_name = 'mean_test_' + metrics[0]
    data_points = len(results[main_metric_name])
    x_values = np.arange(data_points)
    plt.figure(figsize=(20, 10))
    plt.title('Results for ' + estimator, fontsize=16)
    plt.xlabel('Candidates')
    plt.ylabel('Score')
    axe = plt.gca()
    axe.set_xlim(0, data_points - 1)
    axe.set_ylim(0.0, 1.0)

    for metric, color in zip(sorted(metrics), ['g', 'k', 'b', 'r']):
        for sample, style in (('train', '--'), ('test', '-')):
            sample_score_mean = results['mean_%s_%s' % (sample, metric)]
            sample_score_std = results['std_%s_%s' % (sample, metric)]
            axe.fill_between(x_values, sample_score_mean - sample_score_std,
                             sample_score_mean + sample_score_std,
                             alpha=0.1 if sample == 'test' else 0, color=color)
            axe.plot(x_values, sample_score_mean, style, color=color,
                     alpha=1 if sample == 'test' else 0.7,
                     label='%s (%s)' % (metric, sample))

        add_best_metric_value_marker(results, axe, x_values, metric, color)

    plt.legend(loc='best')
    plt.grid(False)
    plt.show()


def get_best_estimator(features, labels, pipelines, cv_strategy, metrics):
    '''
    Get the best estimator from the pipelines, cross validation, metrics and
    refit metrics specified for the search strategy.

    Args:
        features : ndarray
            Array with the features for each data point in the enron dataset.
        labels : ndarray
            Array with the labels for each data point in the enron dataset.
        pipelines : dictionary
            Dictionary with specification of the different pipelines we want to
            use to try and solve this particular problem.
        cv_strategy : cross-validation generator
            Method from the model_selection package that defines a cross
            validation strategy to be used for this particular problem.
        metrics : list
            List containing the different metrics we want to measure for each
            one of the evaluated estimators. The first metric in the list is
            assumed to be the main metric to use for choosing the best
            estimator.

    Returns:
        estimator : Object
            This is the best estimator that was found during the search.
    '''
    print('Performing Model Optimizations...')
    best_main_metric_value = -1.0
    best_estimator = ''
    results = ''
    for estimator, pipeline_definition in pipelines.items():
        print('\nAnalyzing {}...'.format(estimator))
        pipeline = Pipeline(pipeline_definition['pipe'])
        param_grid = pipeline_definition['param_grid']
        clf = GridSearchCV(pipeline, param_grid=param_grid, cv=cv_strategy,
                           scoring=metrics, refit=metrics[0], n_jobs=8,
                           iid=False, verbose=2, return_train_score=True,)
        clf.fit(features, labels)
        results = clf.cv_results_
        print('\nBest {} Found:\n{}\n'.format(estimator, clf.best_estimator_))
        best_estimator_metrics = get_best_estimator_metrics(results, metrics)
        plot_estimator_metrics(estimator, metrics, results)
        if best_estimator_metrics[0] > best_main_metric_value:
            best_estimator = clf.best_estimator_
            best_main_metric_value = best_estimator_metrics[0]

    return results, best_estimator


def get_clean_enron_dataframe(enron_data):
    '''
    Transforms the enron_data dictionary to a DataFrame.

    Args:
        enron_data : dictionary
            Dictionary containing the data stored in the file, in a structured
            format.

    Returns:
        enron_df : DataFrame
            DataFrame containing the data stored in the file, in a structured
            pandas format.
    '''
    # Put the dictionary in a DataFrame and perform some cleaning operations.
    pd.options.display.float_format = '{:20,.2f}'.format
    enron_data.pop('TOTAL', 0)
    enron_df = DataFrame.from_dict(enron_data, orient='index')
    # All NaN strings are converted to Numpy nan values, which allows the
    # describe function to produce proper numeric values for all statistics.
    enron_df.replace('NaN', 0.0, regex=True, inplace=True)
    # Convert True to 1 and False to 0.
    enron_df.replace({True: 1, False: 0}, inplace=True)
    enron_df.drop('email_address', axis=1, inplace=True)

    return enron_df


def plot_features(enron_data):
    '''
    Generate a graphic for each one of the features in a dataframe, in order to
    visualize and help detect easily any outliers present on the data.

    Args:
        enron_data : dictionary
            Dictionary containing the data stored in the file, in a structured
            format.

    Returns:
        enron_df : DataFrame
            This is a secondary behavior, but taking advantage of the call to
            get_clean_enron_dataframe which is required to clean the data
            before plotting it, we return the clean dataframe as a convenience.
            The result is a DataFrame containing the data stored in the file,
            in a structured pandas format.
    '''
    # Plot the variables to understand them better.
    enron_df = get_clean_enron_dataframe(enron_data)
    sns.pairplot(enron_df, hue='poi')

    return enron_df


enron_data = load_data('final_project_dataset.pkl')
enron_feature_list = get_enron_feature_list()
labels, features = get_enron_labels_features(enron_data, enron_feature_list)
labels, features = remove_enron_outliers(labels, features)
labels, features = add_enron_features(labels, features)
pipelines = get_pipelines()
# For cross-validation we'll use a stratified shuffle split because of the
# small size of the dataset.
cv_strategy = StratifiedShuffleSplit(n_splits=10, random_state=42)
# We define all the scoring metrics we want to measure. Recall will be the one
# used to select the best set of parameters, and refit the identifier, because
# in this case false positives are far better than false negatives, since we
# don't want to risk missing ani pois. Recall needs to be the first metric on
# the list, because get_best_estimator assumes the one in that position to be
# the main metric to evaluate the select estimator.
metrics = ['recall', 'accuracy', 'precision', 'f1']
results, best_estimator = get_best_estimator(features, labels, pipelines,
                                             cv_strategy, metrics)
print('\nBest Overall Estimator Found:\n{}\n'.format(best_estimator))

enron_df = plot_features(enron_data)
enron_df.head()
enron_df.describe()

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

# TODO fix this. ¿Maybe refit is needed here before getting results? 
# results = DataFrame.from_dict(best_estimator.cv_results_)
# results.head()

# Task 6: Dump your classifier, dataset, and features_list so anyone can check
# your results. You do not need to change anything below, but make sure that
# the version of poi_id.py that you submit can be run on its own and generates
# the necessary .pkl files for validating your results.

dump_classifier_and_data(best_estimator, enron_data, enron_feature_list)
