# system modules
import unittest

# internal modules
from sdf import ease


class EasingTest(unittest.TestCase):
    def test_min_equals_max(self):
        for e in (ease.linear, ease.in_elastic, ease.in_out_cubic.symmetric):
            with self.subTest(e=e):
                self.assertEqual(e.min.value, -(-e).max.value)
                self.assertEqual(e.min.pos, (-e).max.pos)
