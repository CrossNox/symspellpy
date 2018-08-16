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
    # Spanish keyboard layout aware Damerau OSA
    ES_KLADAMERUAUOSA = 2
    # Portuguese keyboard layout aware Damerau OSA
    PO_KLADAMERUAUOSA = 3


class EditDistance(object):
    def __init__(self, algorithm):
        self._algorithm = algorithm
        if algorithm == DistanceAlgorithm.DAMERUAUOSA:
            self._distance_comparer = DamerauOsa()
        elif algorithm == DistanceAlgorithm.ES_KLADAMERUAUOSA:
            self._distance_comparer = KLADamerauOsa("es")
        elif algorithm == DistanceAlgorithm.PO_KLADAMERUAUOSA:
            self._distance_comparer = KLADamerauOsa("po")
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
        self._base_char = 0
        self._base_char_1_costs = np.zeros(0, dtype=np.int32)
        self._base_prev_char_1_costs = np.zeros(0, dtype=np.int32)

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

        if len_2 > len(self._base_char_1_costs):
            self._base_char_1_costs = np.zeros(len_2, dtype=np.int32)
            self._base_prev_char_1_costs = np.zeros(len_2, dtype=np.int32)
        if max_distance < len_2:
            return self._distance_max(string_1, string_2, len_1, len_2, start,
                                      max_distance, self._base_char_1_costs,
                                      self._base_prev_char_1_costs)
        return self._distance(string_1, string_2, len_1, len_2, start,
                              self._base_char_1_costs,
                              self._base_prev_char_1_costs)

    def _distance(self, string_1, string_2, len_1, len_2, start, char_1_costs,
                  prev_char_1_costs):
        """Internal implementation of the core Damerau-Levenshtein, optimal
        string alignment algorithm.
        from: https://github.com/softwx/SoftWx.Match
        """
        char_1_costs = np.asarray([j + 1 for j in range(len_2)])
        char_1 = " "
        current_cost = 0
        for i in range(len_1):
            prev_char_1 = char_1
            char_1 = string_1[start + i]
            char_2 = " "
            left_char_cost = above_char_cost = i
            next_trans_cost = 0
            for j in range(len_2):
                this_trans_cost = next_trans_cost
                next_trans_cost = prev_char_1_costs[j]
                # cost of diagonal (substitution)
                prev_char_1_costs[j] = current_cost = left_char_cost
                # left now equals current cost (which will be diagonal at
                # next iteration)
                left_char_cost = char_1_costs[j]
                prev_char_2 = char_2
                char_2 = string_2[start + j]
                if char_1 != char_2:
                    # substitution if neither of two conditions below
                    if above_char_cost < current_cost:
                        current_cost = above_char_cost
                    if left_char_cost < current_cost:
                        current_cost = left_char_cost
                    current_cost += 1
                    if (i != 0 and j != 0
                            and char_1 == prev_char_2
                            and prev_char_1 == char_2
                            and this_trans_cost + 1 < current_cost):
                        current_cost = this_trans_cost + 1  # transposition
                char_1_costs[j] = above_char_cost = current_cost
        return current_cost

    def _distance_max(self, string_1, string_2, len_1, len_2, start, max_distance,
                      char_1_costs, prev_char_1_costs):
        """Internal implementation of the core Damerau-Levenshtein, optimal
        string alignment algorithm that accepts a max_distance.
        from: https://github.com/softwx/SoftWx.Match
        """
        char_1_costs = np.asarray([j + 1 if j < max_distance
                                  else max_distance + 1 for j in range(len_2)])
        len_diff = len_2 - len_1
        j_start_offset = max_distance - len_diff
        j_start = 0
        j_end = max_distance
        char_1 = " "
        current_cost = 0
        for i in range(len_1):
            prev_char_1 = char_1
            char_1 = string_1[start + i]
            char_2 = " "
            left_char_cost = above_char_cost = i
            next_trans_cost = 0
            # no need to look beyond window of lower right diagonal -
            # max_distance cells (lower right diag is i - len_diff) and the
            # upper left diagonal + max_distance cells (upper left is i)
            j_start += 1 if i > j_start_offset else 0
            j_end += 1 if j_end < len_2 else 0
            for j in range(j_start, j_end):
                this_trans_cost = next_trans_cost
                next_trans_cost = prev_char_1_costs[j]
                # cost of diagonal (substitution)
                prev_char_1_costs[j] = current_cost = left_char_cost
                # left now equals current cost (which will be diagonal at next
                # iteration)
                left_char_cost = char_1_costs[j]
                prev_char_2 = char_2
                char_2 = string_2[start + j]
                if char_1 != char_2:
                    # substitution if neither of two conditions below
                    if above_char_cost < current_cost:
                        current_cost = above_char_cost
                    if left_char_cost < current_cost:
                        current_cost = left_char_cost
                    current_cost += 1
                    if (i != 0 and j != 0 and char_1 == prev_char_2
                            and prev_char_1 == char_2
                            and this_trans_cost + 1 < current_cost):
                        current_cost = this_trans_cost + 1  # transposition
                char_1_costs[j] = above_char_cost = current_cost
            if char_1_costs[i + len_diff] > max_distance:
                return -1
        return current_cost if current_cost <= max_distance else -1


class KLADamerauOsa(AbstractDistanceComparer):
    def __init__(self, lang="es"):
        if lang == "es":
            self.SL = SpanishLayout()
        elif lang == "po":
            self.SL = PortugueseLayout()
        else:
            raise NotImplementedError("{} keyboard layout hasn't been implemented yet!".format(lang))

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
