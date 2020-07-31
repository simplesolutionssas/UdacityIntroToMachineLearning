#!/usr/bin/python

import matplotlib.pyplot as plt
import importlib
import json
import pandas as pd
from time import time
from itertools import product
from .prep_terrain_data import makeTerrainData
from .class_vis import prettyPicture
# imports required to use different base estimators for AdaBoost experiments
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

features_train, labels_train, features_test, labels_test = makeTerrainData()


def get_points(feature, label_value):
    '''
    Retrieve the points in the features_train dataset that have a specific
    label value for a specific feature.

    Args:
        feature : int
            The column number where the feature data is.
        label_value : int
            The specific label we need to get the points for.

    Returns:
        points : list
            The list of points for the feature that have the desired label.
    '''
    points = [features_train[ii][feature] for ii in range(0,
              len(features_train)) if labels_train[ii] == label_value]

    return points


def visualize_dataset():
    '''
    The training data (features_train, labels_train) has both "fast" and "slow"
    points mixed together, so we separate them by giving them different colors,
    and then produce a scatterplot to visualize them.
    '''
    grade_fast = get_points(0, 0)
    bumpy_fast = get_points(1, 0)
    grade_slow = get_points(0, 1)
    bumpy_slow = get_points(1, 1)

    # initial visualization
    print('dataset visualization')
    plt.xlim(0.0, 1.0)
    plt.ylim(0.0, 1.0)
    plt.scatter(bumpy_fast, grade_fast, color="b", label="fast")
    plt.scatter(grade_slow, bumpy_slow, color="r", label="slow")
    plt.legend()
    plt.xlabel("bumpiness")
    plt.ylabel("grade")
    plt.show()


def get_classifier(full_class_path, **kwargs):
    '''
    Dynamically get an instance of the desired classifier, with the desired
    arguments.

    Args:
        full_class_path : string
            The full path of the class we want to use to classify the data.
            It needs to follow this form: package.subpackage.class.

    Keyword Args:
        These are the specific parameters that we want to use to train the
        classifier. They must be passed as a dictionary with the proper name
        (as required by the classifier class) and the appropriate values. Since
        they vary from class to class, and are so many, it's not reasonable to
        document them here.

    Returns:
        instance : object
            This is the desired classifier, from the class specified by
            full_class_path, instantiated with all the required kwargs.
    '''
    module_name, class_name = full_class_path.rsplit(".", 1)
    class_module = importlib.import_module(module_name)
    classifier_class = getattr(class_module, class_name)
    classifier = classifier_class(**kwargs)
    return classifier


def train(classifier):
    '''
    Get the accuracy of the algorithm, trained with the specific parameters
    defined by the kwargs.

    Args:
        classifier : object
            This is an instance of the class we want to use to classify the
            data, fully initialized with all the desired arguments.

    Returns:
        accuracy : float
            It's the accuracy obtained after classifying the dataset with the
            given classifier.
        training_time : float
            It's the time it took to train the classifier.
    '''
    # fit the model and time it
    start_time = time()
    classifier.fit(features_train, labels_train)
    training_time = round(time()-start_time, 3)
    accuracy = classifier.score(features_test, labels_test)
    return accuracy, training_time


def visualize_decision_boundary(classifier):
    '''
    Visualize the decision boundary on the test data for the given classifier.

    Args:
        classifier : object
            This is the instance of the class we used to classify the data, for
            which we'll plot the decision boundary.
    '''
    try:
        prettyPicture(classifier, features_test, labels_test)
    except NameError:
        pass


def display_best_results(results, results_to_show):
    '''
    Auxiliary method to display all the relevant information for a particular
    number of top results.

    Args:
        results : DataFrame
            Contains the relevant results for all the experiments.
        results_to_show : int
            Is the number of top results that have to be displayed.
    '''
    results.sort_values(by=['accuracy', 'training time'],
                        ascending=[False, True], inplace=True)
    results.reset_index(drop=True, inplace=True)
    for result in range(results_to_show):
        top_result = results.loc[result]
        print('\nclass: {}'.format(top_result['class']))
        print('accuracy: {}'.format(top_result['accuracy']))
        print('parameters: {}'.format(top_result['parameters']))
        print('training time: {} s'.format(top_result['training time']))
        visualize_decision_boundary(top_result['classifier'])


def save_results(results, classifier, accuracy, parameters, training_time):
    '''
    Auxiliary method to save all the relevant information for a particular
    result to the results list.

    Args:
        results : list
            Contains a dictionary per each result, with all relevant results.
        classifier : object
            Instance of the class used to classify the data.
        accuracy : float
            Accuracy obtained after fitting the dataset with the classifier.
        parameters : dictionary
            Contains the parameters used to create and fit the classifier.
        training_time : float
            It's the time it took to train the classifier.
    '''
    result = {}
    result['class'] = type(classifier).__name__
    result['accuracy'] = accuracy
    result['training time'] = training_time
    result['parameters'] = parameters
    result['classifier'] = classifier
    results.append(result)


def create_classifiers(experiment_definitions):
    '''
    Generate the different classifiers that will be used, with all their
    parameters. The parameters definitions for each classifier are defined by
    calculating the Cartesian product of all the possible values that we want
    to test for all parameters for the respective algorithm/classifier class.

    Args:
        experiment_definitions : dictionary (of dictionaries)
            Contains one element for each classification class/
            algorithm that we want to use to try to solve the problem at hand,
            with the respective value being another dictionary that has for
            keys the names of each one of the parameters that we want to tune,
            and as value a list of all the possible levels/values that we want
            this parameter to take.

    Returns:
        classifiers : list
            Contains the different classifier definitions that will be used to
            fit the data an try to solve the classification problem.
    '''
    classifiers = {}
    for classifier, parameters in experiment_definitions.items():
        # solution taken from: https://stackoverflow.com/a/40623158/2316433
        classifier_parameters = [dict(zip(parameters.keys(), values))
                                 for values in product(*parameters.values())]
        classifiers[classifier] = classifier_parameters
    return classifiers


def run_experiments(classifiers):
    '''
    Uses the different classifier definitions stored in classifiers, to execute
    all the experiments and collect their results.

    Args:
        classifiers : list
            Contains the different classifier definitions that will be used to
            fit the data an try to solve the classification problem.

    Results:
        results : list (of dictionaries)
            Contains the relevant results for all the experiments.
    '''
    # create, fit and evaluate each classifier, selecting the best
    results = []
    for classifier_class, kwargs_list in classifiers.items():
        for kwargs in kwargs_list:
            classifier = get_classifier(classifier_class, **kwargs)
            accuracy, training_time = train(classifier)
            save_results(results, classifier, accuracy, kwargs, training_time)

    return results


# define the parameters for all experiments we want to run, in a compact way
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
        }
}
classifiers = create_classifiers(experiment_definitions)
# print('classifiers: \n{}'.format(json.dumps(classifiers, indent=2)))
visualize_dataset()
total_start_time = time()
results = pd.DataFrame(run_experiments(classifiers))
total_training_time = round(time()-total_start_time, 3)
print('\nselection finished. tests executed: {}. total training time: {} s.'
      .format(len(results), total_training_time))
display_best_results(results, 5)
