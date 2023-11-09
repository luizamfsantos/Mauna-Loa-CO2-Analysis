import pandas as pd
import unittest
import re
from datetime import datetime, timedelta
import sys, os
current_dir = os.path.dirname(os.path.realpath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.append(parent_dir)
from src.detrending import polynomial_regression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

class TestPolynomialRegression(unittest.TestCase):
    def setUp(self):
        # Create sample data for testing
        self.X = pd.DataFrame({'x1': [1, 2, 3, 4, 5]})
        self.y = pd.DataFrame({'y': [2, 3.5, 6, 8, 9]})
        self.degree1 = 1
        self.degree2 = 2
        self.degree3 = 3

    def test_polynomial_regression_output(self):
        reg, coef = polynomial_regression(self.X, self.y, self.degree1)
        self.assertIsInstance(reg, LinearRegression)
        self.assertIsInstance(coef, object)

        reg, coef = polynomial_regression(self.X, self.y, self.degree2)
        self.assertIsInstance(reg, LinearRegression)
        self.assertIsInstance(coef, object)

        reg, coef = polynomial_regression(self.X, self.y, self.degree3)
        self.assertIsInstance(reg, LinearRegression)
        self.assertIsInstance(coef, object)

    def test_polynomial_regression_values(self):
        reg, coef = polynomial_regression(self.X, self.y, self.degree1)
        self.assertAlmostEqual(coef[0][0], 1.85, places=2)

        reg, coef = polynomial_regression(self.X, self.y, self.degree2)
        self.assertAlmostEqual(coef[0][0], 2.49, places=2)

        reg, coef = polynomial_regression(self.X, self.y, self.degree3)
        self.assertAlmostEqual(coef[0][0], -1.44, places=2)

    def test_polynomial_regression_type(self):
        reg, coef = polynomial_regression(self.X, self.y, self.degree1)
        self.assertTrue(isinstance(reg, LinearRegression))
        self.assertTrue(isinstance(coef, object))

        reg, coef = polynomial_regression(self.X, self.y, self.degree2)
        self.assertTrue(isinstance(reg, LinearRegression))
        self.assertTrue(isinstance(coef, object))

        reg, coef = polynomial_regression(self.X, self.y, self.degree3)
        self.assertTrue(isinstance(reg, LinearRegression))
        self.assertTrue(isinstance(coef, object))


if __name__ == '__main__':
    unittest.main()
