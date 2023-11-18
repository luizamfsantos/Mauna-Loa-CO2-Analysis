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
        self.test_df = pd.DataFrame({
            't': [0.208333, 0.291667, 0.375, 0.541667, 0.625, 
                  0.708333, 0.875, 0.958333, 1.041667, 1.125],
            'month': [3, 4, 5, 7, 8, 9, 11, 12, 1, 2]
        })

    def test_extract_month_from_t(self):
        # Test if the function returns the expected output
        result_df = extract_month_from_t(self.test_df.copy())  # Making a copy to avoid modifying the original data

        # Asserting equality of the 'month' column in the result with the expected values
        self.assertTrue(result_df['month'].equals(self.test_df['month']), "Months extracted incorrectly")

        # Check if the function raises an AssertionError for an invalid input type
        with self.assertRaises(AssertionError):
            extract_month_from_t("invalid_input")

if __name__ == '__main__':
    unittest.main()
