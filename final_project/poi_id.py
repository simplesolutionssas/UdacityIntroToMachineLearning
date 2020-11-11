#!/usr/bin/python

import sys
import math
import pickle
import pandas as pd
import seaborn as sns
import numpy as np
from time import time
from pandas import DataFrame
from numpy.lib.function_base import average
from matplotlib import pyplot as plt
from sklearn.preprocessing import RobustScaler, MinMaxScaler, Normalizer
from sklearn.decomposition import PCA
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.cluster import KMeans
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, StratifiedShuffleSplit
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.metrics import make_scorer
from backports import tempfile
from tester import dump_classifier_and_data
sys.path.append('../tools/')


def load_data(file_path):
    '''
    Retrieve the dataset stored in the specific pickle file path.

    Args:
        file_path : string
            The absolute file path for the pickle file containing the data.

    Returns:
        dataset : Dictionary
            Dictionary containing the data stored in the file, in a structured
            format.
    '''
    # Load the dictionary containing the dataset
    file = open(file_path, 'r')
    dataset = pickle.load(file)
    file.close()

    return dataset


def get_clean_enron_dataframe(enron_data):
    '''
    Performs cleaning operations on the enron_data_frame.

    Args:
        enron_data : Dictionary
            Dictionary containing the data stored in the file, in a structured
            format.

    Returns:
        enron_data_frame : DataFrame
            DataFrame containing the data stored in the file, in a structured
            pandas format, after cleaning the data.
    '''
    pd.options.display.float_format = '{:20,.2f}'.format
    enron_data_frame = DataFrame.from_dict(enron_data, orient='index')
    # Drop unwanted columns.
    enron_data_frame.drop('TOTAL', axis=0, inplace=True)
    enron_data_frame.drop('email_address', axis=1, inplace=True)
    # All NaN strings are converted to Numpy nan values, which allows the
    # describe function to produce proper numeric values for all statistics.
    enron_data_frame.replace('NaN', np.NaN, regex=True, inplace=True)
    # Convert True to 1 and False to 0.
    enron_data_frame.replace({True: 1, False: 0}, inplace=True)

    return enron_data_frame


def print_missing_values_table(data_frame):
    '''
    Generate a series of statistics for each one of the features found in the
    dataframe in order to understand better the data.

    Adapted from:
    https://www.kaggle.com/willkoehrsen/start-here-a-gentle-introduction

    Args:
        data_frame : DataFrame
            DataFrame we want to inspect for columns with missing values.

    Returns:
        missing_values_table : DataFrame
            DataFrame containing the missing values statistics for the
            data_frame columns.
    '''
    missing_values = data_frame.isna().sum()
    missing_values_percentage = 100 * missing_values / len(data_frame)
    missing_values_table = pd.concat([missing_values,
                                      missing_values_percentage], axis=1)
    # Rename the columns.
    missing_values_table.columns = ['Missing Values', '% of Total Values']
    # Leave on the table only the columns that are missing values.
    columns_missing_values = missing_values_table.iloc[:, 1] != 0
    missing_values_table = missing_values_table[columns_missing_values]
    # Sort the table by percentage of missing descending.
    missing_values_table = missing_values_table.sort_values(
                                '% of Total Values', ascending=False).round(1)
    # Print some summary information.
    print('\nColumns in dataframe: {}.'.format(data_frame.shape[1]))
    print('Columns missing values: {}.'.format(missing_values_table.shape[0]))
    print('\nMissing values table:')
    display(missing_values_table)

    return missing_values_table


def print_target_correlation_report(correlations_table, label_column_name):
    '''
    Generate a report for the most positive and most negative feature
    correlations with the target feature.

    Args:
        data_frame : DataFrame
            DataFrame we want to show correlations with the target feature for.
        correlations_table : DataFrame
            DataFrame containing the correlations between all data features.
        label_column : string
            The name of the column containing the labels for each data point in
            the DataFrame.

    Returns:
        None
    '''
    target_correlations = correlations_table[label_column_name]
    absolute_target_correlations = abs(target_correlations)
    target_correlations_table = pd.concat([target_correlations,
                                           absolute_target_correlations],
                                          axis=1)
    # Rename the columns.
    target_correlations_table.columns = ['Correlation', 'Absolute Correlation']
    # Leave only the features that aren't the target or have a nan value.
    target_correlations_table.drop(label_column_name, axis=0, inplace=True)
    correlation_features = pd.notnull(target_correlations_table.iloc[:, 1])
    target_correlations_table = target_correlations_table[correlation_features]
    # Sort the table by percentage of missing descending.
    target_correlations_table.sort_values('Absolute Correlation',
                                          ascending=False, inplace=True)
    print('\nFeature correlations to ({}) feature:'.format(label_column_name))
    display(target_correlations_table)


def draw_correlation_heatmap(data_frame):
    '''
    Generate a table and heatmap to allow visualization of the correlations
    between input features in the dataset.

    Adapted from:
    https://www.kaggle.com/willkoehrsen/start-here-a-gentle-introduction

    Args:
        data_frame : DataFrame
            DataFrame we want to show correlations for.

    Returns:
        correlations_table : DataFrame
            DataFrame containing the correlations between all data features.
    '''
    correlations_table = data_frame.corr()
    print('\nCorrelation Heatmap:')
    plt.figure(figsize=(16, 12))
    sns.heatmap(correlations_table, cmap='Blues', annot=True)
    plt.title('Correlation Heatmap')
    plt.show()

    return correlations_table


def describe_dataset(data_frame, label_column_name):
    '''
    Generate a series of statistics for each one of the features found in the
    dataframe in order to understand better the data.

    Args:
        data_frame : DataFrame
            DataFrame containing the data stored in the file, in a structured
            format.
        label_column_name : string
            The name of the column containing the labels for each data point in
            the DataFrame.

    Returns:
        None
    '''
    print('\nDataFrame head:')
    display(data_frame)
    print('\nEnron data point count: {}'.format(len(data_frame)))
    print('\nDataFrame info:')
    data_frame.info()
    print('\nDataFrame description:')
    display(data_frame.describe())
    print_missing_values_table(data_frame)
    correlations_table = draw_correlation_heatmap(data_frame)
    print('\nLabel value counts:\n{}'.format(
                                data_frame[label_column_name].value_counts()))
    print_target_correlation_report(correlations_table, label_column_name)


def draw_plot(data_frame, plot_function, label_column_name,
                        plot_columns):
    '''
    Generate a swarm plot for each one of the features in a dataframe, in order
    to visualize and help detect easily any outliers present on the data.

    Args:
        data_frame : DataFrame
            DataFrame containing the data stored in the file, in a structured
            format.
        plot_function : function
            Seaborn function that will be used to plot the data.
        label_column_name : string
            The name of the column containing the labels for each data point in
            the DataFrame.
        plot_columns : integer
            Number of feature plots to display per row.

    Returns:
        None
    '''
    # Separate labels from features for easier plotting.
    labels = data_frame[label_column_name]
    data = data_frame.drop(label_column_name, axis=1)
    # Get the total columns in data, divide it by plot_columns and round it up
    # to get the rows we need to accommodate all features in plot_columns.
    plot_rows = int(math.ceil(float(data.shape[1]) / plot_columns))
    plot_height = plot_rows * 4
    _, axes = plt.subplots(plot_rows, plot_columns, figsize=(16, plot_height))
    figure_count = 0
    print('\nFeature {}:'.format(plot_function.__name__.title()))
    for column in data.columns:
        # Create a dataframe for plotting, with labels and the current column.
        plot_data = pd.concat([labels, data.loc[:, column]], axis=1)
        # Transform the dataframe to the required format using melt.
        plot_data = pd.melt(plot_data, id_vars=label_column_name,
                            var_name=column, value_name='value')
        figure_row = figure_count / plot_columns
        figure_col = figure_count % plot_columns
        figure_count += 1
        ax = axes[figure_row, figure_col]
        plot_function(ax=ax, data=plot_data, hue=label_column_name, x=column,
                      y='value')
        ax.set_xlabel('')
        ax.set_ylabel('')

    plt.show()


def plot_features(data_frame, label_column_name, plot_columns):
    '''
    Generate a graphic for each one of the features in a dataframe, in order to
    visualize and help detect easily any outliers present on the data.

    Args:
        data_frame : DataFrame
            DataFrame containing the data stored in the file, in a structured
            format.
        label_column_name : string
            The name of the column containing the labels for each data point in
            the DataFrame.
        plot_columns : integer
            Number of feature plots to display per row.

    Returns:
        None
    '''
    plot_types = [sns.boxplot, sns.swarmplot, sns.violinplot]
    for plot_type in plot_types:
        draw_plot(data_frame, plot_type, label_column_name, plot_columns)

    # print('\nFeature plots:')
    # plt.figure(figsize=(32, 24))
    # sns.pairplot(features, hue=labels)
    # plt.title('Correlation Heatmap')
    # plt.show()


def remove_enron_outliers(enron_data):
    '''
    Return the labels and features for the Enron dataset, after eliminating the
    outlier data points from the different features.

    Args:
        enron_data : Dictionary
            Dictionary containing the data stored in the file, in a structured
            format.

    Returns:
        enron_data : Dictionary
            Dictionary containing the data after removing the outliers.

    '''
    negatives_removal_features = ['deferral_payments', 'restricted_stock',
                                  'total_stock_value']
    keys = sorted(enron_data.keys())
    removed_outliers = 0
    for key in keys:
        for feature in negatives_removal_features:
            try:
                value = enron_data[key][feature]
                if value < 0:
                    enron_data[key][feature] = 0
                    removed_outliers += 1
            except KeyError:
                print('Error: key {} not present'.format(feature))

    print('\nOutlier features:\n{}'.format(negatives_removal_features))
    print('Total outliers removed:\n{}'.format(removed_outliers))

    return enron_data


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


def get_labels_features(data_dictionary, feature_list):
    """
    Retrieve the labels and features for the given dataset, after applying
    some arranging and cleaning operations:
    - Keys (record IDs) are sorted by alphabetical order.
    - NaN strings are converted to 0.0.
    - Data points where all features have a value of zero are removed.

    Note that the first feature is assumed to be the label feature and is not
    used for determining if the data point should be removed or not.

    Args:
        data_dictionary : Dictionary
            Dictionary containing the data stored in the file, in a structured
            format.
        feature_list : list
            The list of features that needs to be extracted from the dictionary
            and returned for the classification problem. The first feature on
            the list needs to contain the data labels.

    Returns:
        labels : ndarray
            Array with the labels for each data point in the dataset.
        features : ndarray
            Array with the features for each data point in the dataset.
    """
    labels = []
    features = []
    keys = sorted(data_dictionary.keys())
    for key in keys:
        data_point_values = []
        # Get the data point values in a list.
        for feature in feature_list:
            try:
                data_dictionary[key][feature]
            except KeyError:
                print('Error: key {} not present'.format(feature))

            value = data_dictionary[key][feature]
            if value == 'NaN':
                value = 0
            data_point_values.append(float(value))

        # Logic for deciding whether or not to add the data point. The first
        # feature is assumed to be the label feature, and is not considered.
        label_value = data_point_values[0]
        feature_values = data_point_values[1:]
        for value in feature_values:
            if value != 0 and value != 'NaN':
                labels.append(np.array(label_value))
                features.append(np.array(feature_values))
                break

    labels = np.array(labels)
    features = np.array(features)
    print('\nCurrent features and labels shapes:')
    print('Enron labels shape: {}'.format(labels.shape))
    print('Enron features shape: {}'.format(features.shape))

    return labels, features


def get_best_enron_features(labels, features, feature_list, top_n_features):
    '''
    Select the best features to use automatically in a classification problem,
    by using the RandomForestClassifier feature importances.

    Args:
        labels : ndarray
            Array with the labels for each data point in the dataset.
        features : ndarray
            Array with the features for each data point in the dataset.
        feature_list : list
            The list of features that needs to be extracted from the dictionary
            and returned. The first feature is expected to one with the labels.
        top_n_features : integer
            Is the number of features that will be selected from the original
            dataset, according to their importance.

    Returns:
        best_features_list : list
            The list of the best features that will be used for solving the POI
            classification problem.
    '''
    model = RandomForestClassifier(n_estimators=500, n_jobs=8, random_state=42)
    model.fit(features, labels)
    importances = model.feature_importances_
    indices = np.argsort(importances)
    sorted_features = [feature_list[1:][index] for index in indices]
    best_features_list = [feature_list[0]]
    best_features_list.extend(sorted_features[-top_n_features:])
    print('\nSelected features (with label):\n{}'.format(best_features_list))

    # plotting feature importances
    print('\nFeature Importances:')
    plt.figure(figsize=(16, 12))
    plt.title('Feature Importances')
    plt.barh(range(len(indices)), importances[indices], color='b')
    plt.yticks(range(len(indices)), sorted_features)
    plt.xlabel('Relative Importance')
    plt.show()

    return best_features_list


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

    return labels, features


def get_pipelines_definitions():
    '''
    Define the different pipelines that will be used to train and finetune the
    classification model.

    Args:
        None

    Returns:
        pipelines : Dictionary
            A dictionary containing all the pipelines that will be used to fit
            the model in order to select the one that produces the best results
            for the given problem.
    '''
    scale_variations = [None, RobustScaler(), MinMaxScaler(), Normalizer()]
    reduce_dim_variations = [None, PCA(2), PCA(3), PCA(4)]
    pipelines = {
        'GaussianNB': [{
            'classify': [GaussianNB()],
            'scale': scale_variations,
            'reduce_dim': reduce_dim_variations
        }],
        'DecisionTreeClassifier': [{
            'classify': [DecisionTreeClassifier(random_state=42)],
            'scale': scale_variations,
            'reduce_dim': reduce_dim_variations,
            'classify__criterion': ['entropy', 'gini'],
            'classify__splitter': ['best', 'random'],
            'classify__min_samples_split': [2, 4, 8, 16, 32, 64]
        }],
        # I wasn't able to make SVC work with the 'linear' or 'poly' kernels.
        'SVC': [{
                    'classify': [SVC(random_state=42)],
                    'scale': scale_variations,
                    'reduce_dim': reduce_dim_variations,
                    'classify__kernel': ['rbf'],
                    'classify__gamma': ['auto', 'scale'],
                    'classify__C': [10, 100, 1000],
                }, {
                    'classify': [SVC(random_state=42)],
                    'scale': scale_variations,
                    'reduce_dim': reduce_dim_variations,
                    'classify__kernel': ['sigmoid'],
                    'classify__gamma': ['auto', 'scale'],
                    'classify__C': [10, 100, 1000]
                }, {
                    'classify': [SVC(random_state=42)],
                    # With other scalers the search won't finish.
                    'scale': [None, MinMaxScaler()],
                    # With values over 6 (8, 16, None) the search won't finish.
                    'reduce_dim': [PCA(2), PCA(4)],
                    'classify__kernel': ['poly'],
                    'classify__gamma': ['auto', 'scale'],
                    'classify__C': [10, 100, 1000],
                    # With a value of 2 or 3 the search won't finish.
                    'classify__degree': [4, 5]
            }],
        'KNeighborsClassifier': [{
            'classify': [KNeighborsClassifier()],
            'scale': scale_variations,
            'reduce_dim': reduce_dim_variations,
            'classify__n_neighbors': [2, 4, 8, 16, 32],
            'classify__weights': ['uniform', 'distance'],
            'classify__algorithm': ['ball_tree', 'kd_tree', 'brute'],
            'classify__p': [1, 2]
        }],
        'RandomForestClassifier': [{
            'classify': [RandomForestClassifier(random_state=42)],
            'scale': scale_variations,
            'reduce_dim': reduce_dim_variations,
            'classify__n_estimators': [4, 8, 16],
            'classify__criterion': ['entropy', 'gini'],
            'classify__min_samples_split': [4, 8, 16],
            'classify__max_depth': [4, 8, 16],
            'classify__max_features': [None, 'sqrt', 'log2']
        }],
        'AdaBoostClassifier': [{
                'classify': [AdaBoostClassifier(random_state=42)],
                'scale': scale_variations,
                'reduce_dim': reduce_dim_variations,
                'classify__base_estimator': [
                    None,
                    SVC(kernel='poly', gamma='scale', degree=5),
                    DecisionTreeClassifier(splitter='random')
                ],
                'classify__n_estimators': [32, 64, 128],
                'classify__algorithm': ['SAMME'],
                'classify__learning_rate': [0.05, 0.1, 0.3, 1]
        }],
        'KMeans': [{
                'classify': [KMeans(random_state=42)],
                'scale': scale_variations,
                'reduce_dim': reduce_dim_variations,
                'classify__n_clusters': [2]
        }]
    }

    return pipelines


def get_dummy_pipeline_with_memory():
    '''
    Return a pipeline to be used in a search strategy (e.g. GridSearchCV,
    RandomSearchCV, etc.), with the correct steps in the right sequence, but
    initialized with arbitrary estimators (because the specific estimators
    to use in the search will be defined by means of the param_grid).

    The returned pipeline uses memory to improve search performance.

    Args:
        None

    Returns:
        pipeline : Pipeline
            A Pipeline object with the desired steps in the proper sequence,
            but initialized with arbitrary estimators, and with memory usage
            enabled.
    '''
    with tempfile.TemporaryDirectory(prefix='poi_id_') as tmpdir:
        # The steps used are just for initializing the pipeline. The actual
        # steps are defined inside the param_grid.
        pipeline = Pipeline(steps=[('scale', RobustScaler()),
                                   ('reduce_dim', PCA()),
                                   ('classify', GaussianNB())],
                            memory=tmpdir)

    return pipeline


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
        metrics : list
            List containing the names of the metrics evaluated for the
            estimator during the search. The first metric in the list is
            assumed to be the main metric, which was used to select the best
            estimator.
        results : DataFrame
            DataFrame with the results of the estimator's grid search.

    Returns:
        None
    '''
    # TODO explore with a pivot table in pandas that gets the average metric
    # score (for all metrics evaluated) for each value of a parameter. This
    # could be then plotted to see the impact of the specific parameter on
    # the results. Iterating over the different parameters we would end up
    # with a group of charts (one per parameter) to detect those parameters
    # most important for solving the particular problem.
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
        pipelines : Dictionary
            Dictionary with specification of the different pipelines we want to
            use to try and solve this particular problem.
        cv_strategy : cross-validation generator
            Method from the model_selection package that defines a cross
            validation strategy to be used for this particular problem.
        metrics : Dictionary
            Dictionary containing the names of the different metrics we want to
            measure for each one of the evaluated estimators. The first metric
            in the Dictionary is assumed to be the main metric to use for
            choosing the best estimator.

    Returns:
        best_results : DataFrame
            DataFrame with the best results of the grid search.
        best_estimator : Object
            This is the best estimator that was found during the search.
    '''
    metric_names = list(metrics.keys())
    print('\nPerforming Model Optimizations...')
    best_main_metric_value = -1.0
    best_estimator = ''
    best_results = ''
    pipeline = get_dummy_pipeline_with_memory()
    for estimator, pipeline_definition in pipelines.items():
        print('\nAnalyzing {}...'.format(estimator))
        clf = GridSearchCV(pipeline, param_grid=pipeline_definition,
                           cv=cv_strategy, scoring=metrics,
                           refit=metric_names[0], iid=False,
                           n_jobs=8, verbose=True, error_score='raise',
                           return_train_score=True)
        clf.fit(features, labels)
        results = clf.cv_results_
        print('\nBest {} Found:\n{}\n'.format(estimator, clf.best_estimator_))
        best_estimator_metrics = get_best_estimator_metrics(results,
                                                            metric_names)
        plot_estimator_metrics(estimator, metric_names, results)
        if best_estimator_metrics[0] > best_main_metric_value:
            best_estimator = clf.best_estimator_
            best_results = results
            best_main_metric_value = best_estimator_metrics[0]

    return best_results, best_estimator


def custom_score(labels, predictions):
    '''
    Calculate the score for the predictions, based on the labels passed to the
    function, using a combination of accuracy, recall and precision, in an
    attempt to get a model with a good accuracy and good enough precision and
    recall values.

    Args:
        labels : ndarray
            Array with the labels for each data point in the dataset.
        predictions : ndarray
            Array with the predictions for each data point in the dataset.

    Returns:
        total_score : double
            The score assigned to the model, given the predictions.
    '''
    accuracy = accuracy_score(labels, predictions)
    precision = precision_score(labels, predictions)
    recall = recall_score(labels, predictions)
    # accuracy is included twice in the average, to give increase its weight.
    total_score = average([accuracy, accuracy, precision, recall])

    return total_score


def print_overall_results(start_time, results, metrics, best_estimator):
    '''
    Print the best estimator with the respective metrics and other information.

    Args:
        start_time : float
            The time when the search process started.
        results : DataFrame
            DataFrame with the results of the estimator's grid search.
        metrics : list
            List containing the names of the metrics evaluated for the
            estimator during the search.
        best_estimator : string
            The definition of the best estimator found.

    Returns:
        None
    '''
    training_time = round(time() - start_time, 3)
    print('\nTotal training time: {} s'.format(training_time))
    print('\nBest Overall Results:')
    get_best_estimator_metrics(results, list(metrics.keys()))
    print('\nBest Overall Estimator Found:\n{}'.format(best_estimator))


# Task 0: Load and explore the dataset and features.
enron_data = load_data('final_project_dataset.pkl')
enron_data_frame = get_clean_enron_dataframe(enron_data)
describe_dataset(enron_data_frame, 'poi')
plot_features(enron_data_frame, 'poi', 3)

# Task 1: Remove outliers
enron_data = remove_enron_outliers(enron_data)

# Task 2: Select what features you'll use.
full_enron_feature_list = get_enron_feature_list()
labels, features = get_labels_features(enron_data, full_enron_feature_list)
enron_feature_list = get_best_enron_features(labels, features,
                                             full_enron_feature_list, 8)
labels, features = get_labels_features(enron_data, enron_feature_list)
labels, features = add_enron_features(labels, features)

# TODO Task 3: Create new feature(s)

# Task 4: Try a variety of classifiers
# Please name your classifier clf for easy export below.
# Note that if you want to do PCA or other multi-stage operations,
# you'll need to use Pipelines. For more info:
# http://scikit-learn.org/stable/modules/pipeline.html
pipelines = get_pipelines_definitions()

# Task 5: Tune your classifier to achieve better than .3 precision and recall
# using our testing script. Check the tester.py script in the final project
# folder for details on the evaluation method, especially the test_classifier
# function. Because of the small dataset size, the test script uses stratified
# shuffle split cross validation, so that's what we'll use here as well.
# For more info:
# http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html
cv_strategy = StratifiedShuffleSplit(n_splits=10, random_state=42)
# We define all the scoring metrics we want to measure. Recall will be the one
# used to select the best set of parameters, and refit the identifier, because
# in this case false positives are far better than false negatives, since we
# don't want to risk missing ani pois. Recall needs to be the first metric on
# the list, because get_best_estimator assumes the one in that position to be
# the main metric to evaluate the select estimator.
start_time = time()
# To guarantee dictionary order, we pass an iterable of key-value pairs.
metrics = OrderedDict([
    ('overall', make_scorer(custom_score)),
    ('accuracy', 'accuracy'),
    ('recall', 'recall'),
    ('precision', 'precision'),
])
results, best_estimator = get_best_estimator(features, labels, pipelines,
                                             cv_strategy, metrics)
print_overall_results(start_time, results, metrics, best_estimator)

# TODO fix this. ¿Maybe refit is needed here before getting results?
# results = DataFrame.from_dict(best_estimator.cv_results_)
# results.head()

# Task 6: Dump your classifier, dataset, and features_list so anyone can check
# your results. You do not need to change anything below, but make sure that
# the version of poi_id.py that you submit can be run on its own and generates
# the necessary .pkl files for validating your results.
dump_classifier_and_data(best_estimator, enron_data, enron_feature_list)
