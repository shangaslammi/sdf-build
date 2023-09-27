import math
import numpy as np

pi = math.pi

degrees = math.degrees
radians = math.radians


def n_trailing_ascending_positive(d):
    """
    Determine how many elements in a given sequence are positive and ascending.

    Args:
        d (sequence of numbers): the sequence to check

    Returns:
        int : the amount of trailing ascending positive elements
    """
    d = np.array(d).flatten()
    # is the next element larger than previous and positive?
    order = (d[1:] > d[:-1]) & (d[:-1] > 0)
    # TODO: Not happy at all with this if/else mess. Is there no easier way to find the
    #       index in a numpy array after which the values are only ascending? 🤔
    if np.all(order):  # all ascending
        return d.size
    elif np.all(~order):  # none ascending
        return 0
    else:  # count from end how many are ascending
        return np.argmin(order[::-1]) + 1
