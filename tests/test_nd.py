import unittest
import numpy as np
from sdf import *


class NdTest(unittest.TestCase):
    def test_distance_to_plane(self):
        self.assertEqual(dn.distance_to_plane(p=ORIGIN, origin=ORIGIN, normal=Z), 0)
        self.assertEqual(dn.distance_to_plane(p=ORIGIN, origin=-3 * Z, normal=Z), 3)
        self.assertEqual(dn.distance_to_plane(p=ORIGIN, origin=2 * Z, normal=Z), 2)
        np.testing.assert_equal(
            dn.distance_to_plane(p=np.array([X, Y, Z]), origin=2 * Z, normal=10 * Z),
            np.array([2, 2, 1]),
        )
        np.testing.assert_equal(
            dn.distance_to_plane(
                p=np.array([X + Y, Y - Z, 2 * Z - 5 * X]), origin=X, normal=-3 * X
            ),
            np.array([0, 1, 6]),
        )
