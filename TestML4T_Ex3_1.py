import unittest
import ML4T_Ex3_1
import pandas as pd


if __name__ == '__main__':
    unittest.main()


class TestML4T_Ex3_1(unittest.TestCase):
    def test_calc_entropy(self):
        a = pd.DataFrame([[0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0]])
        result = ML4T_Ex3_1.calc_entropy(a)
        self.assertAlmostEqual(result, 0.94028595867063092)

    def test_calc_feature_entropy(self):
        a = pd.DataFrame([[1, 1], [1, 1], [1, 1], [1, 1], [1, 1], [1, 1], [0, 1], [0, 1], [0, 1], [1, 0], [1, 0], [0, 0], [0, 0], [0, 0]], columns=['feature', 'result'])
        result = ML4T_Ex3_1.calc_feature_entropy(a, 'feature')
        print(result)
        self.assertAlmostEqual(result, 0.0481270304083)