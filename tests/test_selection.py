import unittest

from ekaquant.selection import threshold_cumulative, threshold_gradient, threshold_pct


class TestSelectionThresholds(unittest.TestCase):
    def test_threshold_pct(self):
        value = threshold_pct({"a": 1.0, "b": 2.0, "c": 3.0, "d": 4.0}, 0.25)
        self.assertAlmostEqual(value, 3.25, places=6)

    def test_threshold_gradient_single(self):
        value = threshold_gradient({"a": 1.0}, sensitivity_ratio=0.05)
        self.assertEqual(value, 1.0)

    def test_threshold_cumulative_zero(self):
        value = threshold_cumulative({"a": 0.0, "b": 0.0}, budget=0.95)
        self.assertEqual(value, 0.0)


if __name__ == "__main__":
    unittest.main()
