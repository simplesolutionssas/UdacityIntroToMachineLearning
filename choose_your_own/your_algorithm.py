#!/usr/bin/python

import matplotlib.pyplot as plt
import importlib
from time import time
from prep_terrain_data import makeTerrainData
from class_vis import prettyPicture

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


def display_results(classifier, accuracy, parameters, training_time):
    '''
    Auxiliary method to display all the relevant information for a particular
    result.

    Args:
        classifier : object
            This is the instance of the class we used to classify the data, for
            which we'll display the results.
        accuracy : float
            It's the accuracy obtained after classifying the dataset with the
            given classifier.
        kwargs : dictionary
            This is a dictionary with the particular parameters used to create
            and fit the classifier that obtained these results.
        training_time : float
            It's the time it took to train the classifier.
    '''
    print('\nclass: {}'.format(type(classifier).__name__))
    print('\accuracy: {}'.format(accuracy))
    print('kwargs: {}'.format(kwargs))
    print('training time: {} s'.format(training_time))
    visualize_decision_boundary(classifier)


def unpack_parameters(parameters_list):
    '''
    '''
    parameters = []
    for parameter_name, parameter_values in parameters_list.items():
        if len(parameters) == 0:
            parameters = [{parameter_name: parameter_value}
                          for parameter_value in parameter_values]
        else:
            parameters = [dict(parameter.items() +
                               {parameter_name: parameter_value}.items())
                          for parameter_value in parameter_values
                          for parameter in parameters]

    return parameters


def create_classifiers(experiment_definitions):
    '''
    '''
    classifiers = {}
    for classifier_class, parameters_list in experiment_definitions.items():
        classifiers[classifier_class] = unpack_parameters(parameters_list)
    return classifiers


# define the parameters for all experiments we want to run, in a compact way
experiment_definitions = {
    'sklearn.tree.DecisionTreeClassifier':
        {'criterion': ['entropy', 'gini'],
         'min_samples_split': [2, 4, 8, 16, 32, 64]}
}
classifiers = create_classifiers(experiment_definitions)
print(classifiers)

# create, fit and evaluate each classifier, selecting the best
max_accuracy = 0
visualize_dataset()
for classifier_class, kwargs_list in classifiers.items():
    for kwargs in kwargs_list:
        classifier = get_classifier(classifier_class, **kwargs)
        accuracy, training_time = train(classifier)
        if accuracy > max_accuracy:
            max_accuracy = accuracy
            display_results(classifier, accuracy, kwargs, training_time)

print('selection finished')
