from enum import Enum

import numpy as np

import symspellpy.helpers as helpers
from symspellpy.keyboards import *


class DistanceAlgorithm(Enum):
    """Supported edit distance algorithms"""
    # Levenshtein algorithm.
    LEVENSHTEIN = 0
    # Damerau optimal string alignment algorithm
    DAMERUAUOSA = 1


class EditDistance(object):
    def __init__(self, algorithm):
        self._algorithm = algorithm
        if algorithm == DistanceAlgorithm.DAMERUAUOSA:
            self._distance_comparer = DamerauOsa()
        else:
            raise ValueError("Unknown distance algorithm")

    def compare(self, string_1, string_2, max_distance):
        """Compare a string to the base string to determine the edit distance,
        using the previously selected algorithm.

        Keyword arguments:
        string_1 -- Base string.
        string_2 -- The string to compare.
        max_distance -- The maximum distance allowed.

        Return:
        The edit distance (or -1 if max_distance exceeded).
        """
        return self._distance_comparer.distance(string_1, string_2, max_distance)


class AbstractDistanceComparer(object):
    def distance(self, string_1, string_2, max_distance):
        """Return a measure of the distance between two strings.

        Keyword arguments:
        string_1 -- One of the strings to compare.
        string_2 -- The other string to compare.
        max_distance -- The maximum distance that is of interest.

        Return:
        -1 if the distance is greater than the max_distance,
        0 if the strings are equivalent, otherwise a positive number whose
        magnitude increases as difference between the strings increases.
        """
        raise NotImplementedError("Should have implemented this")


class DamerauOsa(AbstractDistanceComparer):
    def __init__(self):
        self.SL = SpanishLayout()

    def distance(self, string_1, string_2, max_distance):
        if string_1 is None or string_2 is None:
            return helpers.null_distance_results(string_1, string_2,
                                         max_distance)
        if max_distance <= 0:
            return 0 if string_1 == string_2 else -1
        max_distance = int(min(2 ** 31 - 1, max_distance))
        # if strings of different lengths, ensure shorter string is in string_1.
        # This can result in a little faster speed by spending more time
        # spinning just the inner loop during the main processing.
        if len(string_1) > len(string_2):
            string_2, string_1 = string_1, string_2
        if len(string_2) - len(string_1) > max_distance:
            return -1
        # identify common suffix and/or prefix that can be ignored
        len_1, len_2, start = helpers.prefix_suffix_prep(string_1, string_2)
        if len_1 == 0:
            return len_2 if len_2 <= max_distance else -1
        return self._distance_max(string_1[start:], string_2[start:], len_1, len_2, max_distance)

    def _distance_max(self, string_1, string_2, len_1, len_2, max_distance):
        weights = np.zeros((len_1 + 1, len_2 + 1))
        weights[:, 0] = np.array(range(len_1 + 1))
        weights[0, :] = np.array(range(len_2 + 1))
        for i in range(1, len_1 + 1):
            for j in range(1, len_2 + 1):
                if string_1[i - 1] == string_2[j - 1]:
                    cost = 0
                else:
                    cost = self.SL.sustitution_weight(
                        string_1[i - 1], string_2[j - 1])
                weights[i, j] = min([
                    weights[i - 1, j] + 1,  # delete
                    weights[i, j - 1] + 1,  # insert
                    weights[i - 1, j - 1] + cost  # substitution
                ])
                if i > 1 and j > 1 and string_1[i - 2] == string_2[j - 1] and string_1[i - 1] == string_2[j - 2]:
                    weights[i, j] = min([
                        weights[i, j],
                        weights[i - 3, j - 3] + cost  # transposition
                    ])
            if not any(weights[i, :] <= max_distance):
                return -1
        return weights[-1, -1] if weights[-1, -1] <= max_distance else -1
