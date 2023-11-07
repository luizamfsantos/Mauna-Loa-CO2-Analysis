import pandas as pd
import unittest
import re
from datetime import datetime, timedelta
import sys, os
current_dir = os.path.dirname(os.path.realpath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.append(parent_dir)
from src.preprocessing import remove_non_numeric_chars, add_time_index, test_train_split

class TestRemoveNonNumericChars(unittest.TestCase):

    def test_only_digits(self):
        self.assertEqual(remove_non_numeric_chars('123'), '123')

    def test_with_negative_sign(self):
        self.assertEqual(remove_non_numeric_chars('-123'), '-123')

    def test_with_mixed_chars(self):
        self.assertEqual(remove_non_numeric_chars('a1b2c3'), '123')

    def test_with_float(self):
        self.assertEqual(remove_non_numeric_chars('1.23'), '1.23')

    def test_with_special_chars(self):
        self.assertEqual(remove_non_numeric_chars('!@#$%^&*()_+-=[]{}|;:,<>/?'), '-')

    def test_with_empty_string(self):
        self.assertEqual(remove_non_numeric_chars(''), '')

    def test_with_no_digits(self):
        self.assertEqual(remove_non_numeric_chars('abc'), '')
    
    def test_with_leading_and_trailing_spaces(self):
        self.assertEqual(remove_non_numeric_chars(' 123 '), '123')


class TestAddTimeIndex(unittest.TestCase):

    def setUp(self):
        # Create sample DataFrame for testing
        dates = [datetime(1958, 1, 1), datetime(1958, 2, 17), datetime(1958, 3, 15)]
        self.df = pd.DataFrame({'exact_date': dates})

    def test_valid_input(self):
        result_df = add_time_index(self.df)
        self.assertTrue('t' in result_df.columns, "Column 't' not found in the DataFrame")
        self.assertTrue(all(isinstance(val, (int, float)) for val in result_df['t']), "Column 't' does not contain numeric values")

    def test_invalid_input_df(self):
        with self.assertRaises(AssertionError, msg="Expected AssertionError for invalid df"):
            add_time_index("invalid_df")

    def test_invalid_input_date_col(self):
        with self.assertRaises(AssertionError, msg="Expected AssertionError for invalid date_col"):
            add_time_index(self.df, date_col=123)

    def test_invalid_column_name(self):
        with self.assertRaises(AssertionError, msg="Expected AssertionError for invalid column name"):
            add_time_index(self.df, date_col='invalid_date_col_name')
    
    def test_specific_date(self):
        specific_date = datetime(1959, 1, 17)
        expected_time_index = 12.5
        self.df = pd.DataFrame({'exact_date': [specific_date]})
        result_df = add_time_index(self.df)
        calculated_time_index = result_df['t'].iloc[0]
        self.assertEqual(calculated_time_index, expected_time_index, f"Incorrect time index for {specific_date}. Expected {expected_time_index}, but got {calculated_time_index}")


class TestTrainSplit(unittest.TestCase):
    def setUp(self):
        # Create a sample DataFrame for testing
        self.df = pd.DataFrame({'A': [1, 2, 3, 4, 5], 'B': [6, 7, 8, 9, 10]})

    def test_test_train_split(self):
        df_train, df_test = test_train_split(self.df)
        
        # Check if the length of the training data is 80% of the original DataFrame
        self.assertEqual(len(df_train), int(len(self.df) * 0.8))
        
        # Check if the length of the test data is 20% of the original DataFrame
        self.assertEqual(len(df_test), len(self.df) - int(len(self.df) * 0.8))
        
        # Check if the first part of the original DataFrame is the same as the training data
        self.assertTrue(self.df.iloc[:int(len(self.df) * 0.8)].equals(df_train))
        
        # Check if the second part of the original DataFrame is the same as the test data
        self.assertTrue(self.df.iloc[int(len(self.df) * 0.8):].equals(df_test))


if __name__ == '__main__':
    unittest.main()
