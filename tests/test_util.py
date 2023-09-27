import unittest
import itertools
from sdf import *
import numpy as np


class TestUtils(unittest.TestCase):
    def test_n_trailing_ascending_positive(self):
        f = util.n_trailing_ascending_positive
        self.assertEqual(f([1, 2, 3, 4]), 4)
        self.assertEqual(f([1, 2, 3, 2, 4]), 2)
        self.assertEqual(f([-1, -2, -4]), 0)
        self.assertEqual(f([]), 0)
        self.assertEqual(f([1]), 1)
        self.assertEqual(f([1, 2]), 2)
        self.assertEqual(f([2, 1]), 0)
        self.assertEqual(f([2, 1, 2, 3, 4]), 4)
        self.assertEqual(f([2, 1, 2, -3, 4]), 1)
