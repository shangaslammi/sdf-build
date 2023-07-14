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
        for axis, angle, point, rotated in [
            (Z, units("90°"), [1, 0, 0], [0, 1, 0]),
            (X, units("90°"), [1, 0, 0], [1, 0, 0]),
            (Z, units("45°"), [1, 0, 0], [0.707107, 0.707107, 0]),
            (Y, units("45°"), [1, 0, 0], [0.707107, 0, -0.707107]),
            (Y, units("-45°"), [1, 0, 0], [0.707107, 0, 0.707107]),
        ]:
            with self.subTest(axis=axis, angle=angle, point=point, rotated=rotated):
                np.testing.assert_allclose(
                    rotation_matrix(axis=axis, angle=angle) @ np.array(point),
                    rotated,
                    err_msg="matrix product with @ does not work",
                    atol=1e-5,
                    rtol=1e-5,
                ),
                np.testing.assert_allclose(
                    np.dot(rotation_matrix(axis=axis, angle=angle), np.array(point)),
                    rotated,
                    err_msg="dot product does not work",
                    atol=1e-5,
                    rtol=1e-5,
                )
                np.testing.assert_allclose(
                    np.dot(
                        np.array(point),
                        rotation_matrix(axis=axis, angle=-angle),
                    ),
                    rotated,
                    err_msg="swapped dot product with negative angle does not work",
                    atol=1e-5,
                    rtol=1e-5,
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
