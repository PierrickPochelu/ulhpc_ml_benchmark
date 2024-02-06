import unittest
from src.bench import bench

class MyTestCase(unittest.TestCase):
    def test_cls(self):
        scores=bench(num_samples=10, num_features=2, fix_comp_time=0.01, reg_or_cls="cls")
        self.assertGreater(len(scores),10)

        algo_names = set([s[0] for s in scores])

        self.assertIn("GaussianProcessClassifier", algo_names)
        self.assertIn("MLPClassifier", algo_names)
        self.assertIn("DecisionTreeClassifier", algo_names)

        self.assertNotIn("KNeighborsRegressor", algo_names)
        self.assertNotIn("Ridge", algo_names)
        self.assertNotIn("MLPRegressor", algo_names)

    def test_reg(self):
        scores = bench(num_samples=10, num_features=2, fix_comp_time=0.01, reg_or_cls="reg")
        self.assertGreater(len(scores),10)

        algo_names = set([s[0] for s in scores])

        self.assertIn("KNeighborsRegressor", algo_names)
        self.assertIn("Ridge", algo_names)
        self.assertIn("MLPRegressor", algo_names)

        self.assertNotIn("GaussianProcessClassifier", algo_names)
        self.assertNotIn("MLPClassifier", algo_names)
        self.assertNotIn("DecisionTreeClassifier", algo_names)