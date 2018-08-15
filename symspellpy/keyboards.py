import numpy as np


class KeyboardLayout():

    _x_weight = 1.4
    _y_weight = 1.0
    _weight_matrix = np.diag([_x_weight, _y_weight])

    def sustitution_weight(self, a, b):
        pos_a = np.array(type(self)._dictionary_.get(a))
        pos_b = np.array(type(self)._dictionary_.get(b))
        try:
            pos_a = np.dot(KeyboardLayout._weight_matrix, pos_a)
            pos_b = np.dot(KeyboardLayout._weight_matrix, pos_b)
            return np.linalg.norm(pos_a - pos_b)
        except:
            return 1


class SpanishLayout(KeyboardLayout):
    _dictionary_ = {'q': (0, 0), 'w': (0, 1), 'e': (0, 2), 'r': (0, 3), 't': (0, 4), 'y': (0, 5), 'u': (0, 6), 'i': (0, 7), 'o': (0, 8), 'p': (0, 9), 'a': (1, 0), 's': (1, 1), 'd': (1, 2), 'f': (
        1, 3), 'g': (1, 4), 'h': (1, 5), 'j': (1, 6), 'k': (1, 7), 'l': (1, 8), 'ñ': (1, 9), 'z': (2, 1.5), 'x': (2, 2.5), 'c': (2, 3.5), 'v': (2, 4.5), 'b': (2, 5.5), 'n': (2, 6.5), 'm': (2, 7.5)}


class PortugueseLayout(KeyboardLayout):
    _dictionary_ = {'q': (0, 0), 'w': (0, 1), 'e': (0, 2), 'r': (0, 3), 't': (0, 4), 'y': (0, 5), 'u': (0, 6), 'i': (0, 7), 'o': (0, 8), 'p': (0, 9), 'a': (1, 0), 's': (1, 1), 'd': (1, 2), 'f': (
        1, 3), 'g': (1, 4), 'h': (1, 5), 'j': (1, 6), 'k': (1, 7), 'l': (1, 8), 'ç': (1, 9), 'z': (2, 1.5), 'x': (2, 2.5), 'c': (2, 3.5), 'v': (2, 4.5), 'b': (2, 5.5), 'n': (2, 6.5), 'm': (2, 7.5)}
