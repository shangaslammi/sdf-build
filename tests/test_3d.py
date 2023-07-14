import unittest
import itertools
from sdf import *

import numpy as np


class TestRotation(unittest.TestCase):
    def test_rotation_matrix_roundtrip(self):
        np.random.seed(42)
        for angle in np.arange(-360, 360, 45):
            for axis in np.random.uniform(-10, 10, (10, 3)):
                with self.subTest(angle=angle, axis=axis):
                    points = np.random.uniform(-10, 10, (10, 3))
                    matrix = rotation_matrix(axis=axis, angle=angle)
                    points_rotated = np.dot(points, matrix)
                    matrix_back = rotation_matrix(axis=axis, angle=-angle)
                    rotated_back = np.dot(points_rotated, matrix_back)
                    self.assertTrue(
                        np.allclose(rotated_back, points),
                        "Rotating back and forth changes stuff!",
                    )

    def test_rotation_matrix(self):
        np.testing.assert_allclose(
            np.dot([1, 0, 0], rotation_matrix(axis=Z, angle=units("90°"))),
            [0, 1, 0],
        )


class Test3D(unittest.TestCase):
    def test_orient(self):
        directions = set(
            tuple(np.sum(c, axis=0))
            for c in itertools.chain.from_iterable(
                itertools.combinations(
                    [a * b for a, b in itertools.product([X, Y, Z], [1, -1])], r=r
                )
                for r in range(1, 4)
            )
        )
        for length in [10, 20]:
            for radius in [5, 25]:
                for direction in [np.array(d) for d in directions]:
                    for scale in [1, 5]:
                        direction *= scale
                        if np.allclose(direction, 0):
                            continue
                        with self.subTest(
                            length=length, radius=radius, direction=direction
                        ):
                            c = capsule(ORIGIN, length * Z, radius=radius)
                            self.assertAlmostEqual(c([ORIGIN]).flatten()[0], -radius)
                            self.assertAlmostEqual(
                                c([length * Z]).flatten()[0], -radius
                            )
                            o = c.orient(direction)
                            self.assertAlmostEqual(c([ORIGIN]).flatten()[0], -radius)
                            self.assertAlmostEqual(
                                o(
                                    [length * direction / np.linalg.norm(direction)]
                                ).flatten()[0],
                                -radius,
                                msg=f"orient({direction}) doesn't work",
                            )
