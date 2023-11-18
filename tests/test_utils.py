import pandas as pd
import unittest
import sys, os
current_dir = os.path.dirname(os.path.realpath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.append(parent_dir)
from src.utils import extract_month_from_t

class TestExtractMonthFromT(unittest.TestCase):

    def setUp(self):
        # Test data within setUp to be available for all test methods
        self.test_values = [0.208333, 0.291667, 0.375, 0.541667, 0.625,
                            0.708333, 0.875, 0.958333, 1.041667, 1.125]
        self.expected_months = [3, 4, 5, 7, 8, 9, 11, 12, 1, 2]

    def test_extract_month_from_t(self):
        # Test if the function returns the expected output for each test value
        for i, test_value in enumerate(self.test_values):
            result_month = extract_month_from_t(test_value)
            expected_month = self.expected_months[i]

            self.assertEqual(result_month, expected_month, f"Month extracted incorrectly for value: {test_value}")

        # Check if the function raises an AssertionError for an invalid input type
        with self.assertRaises(TypeError):
            extract_month_from_t("invalid_input")


if __name__ == '__main__':
    unittest.main()
