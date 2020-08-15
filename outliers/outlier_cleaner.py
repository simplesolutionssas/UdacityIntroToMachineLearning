#!/usr/bin/python

import numpy as np


def outlierCleaner(predictions, ages, net_worths):
    """
        Clean away the 10% of points that have the largest
        residual errors (difference between the prediction
        and the actual net worth).

        Return a list of tuples named cleaned_data where
        each tuple is of the form (age, net_worth, error).
    """
    cleaned_data = []
    # your code goes here
    errors = abs(net_worths - predictions)
    points_to_clean = int(len(predictions) * 0.1)
    points_to_keep = len(predictions) - points_to_clean
    # this function returns an unordered array of indices where the point at
    # the points_to_keep position is in the order it would be if the array
    # was ordered, and all elements before it are smaller than that, and all
    # elements that are bigger come after it (although the particular order on
    # the two partitions isn't guaranteed). we then slice this array, getting
    # only the first points_to_keep, which are the array's smaller elements.
    indices_to_keep = np.argpartition(errors, points_to_keep,
                                      axis=0)[:points_to_keep, 0]
    cleaned_data = [(ages[index], net_worths[index], errors[index])
                    for index in indices_to_keep]

    return cleaned_data
