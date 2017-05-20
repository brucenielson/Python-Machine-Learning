import unittest
import ML4T_Ex3_1
import pandas as pd


if __name__ == '__main__':
    unittest.main()


class TestML4T_Ex3_1(unittest.TestCase):
    def test_calc_entropy(self):
        data = pd.DataFrame([[0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0]])
        entropy = ML4T_Ex3_1.calc_entropy(data)
        self.assertAlmostEqual(entropy, 0.94028595867063092)

    def test_calc_info_gain(self):
        data = pd.DataFrame([[1, 1], [1, 1], [1, 1], [1, 1], [1, 1], [1, 1], [0, 1], [0, 1], [0, 1], [1, 0], [1, 0], [0, 0], [0, 0], [0, 0]], columns=['feature', 'result'])
        gain, best_value = ML4T_Ex3_1.calc_info_gain(ML4T_Ex3_1.calc_entropy, data, 'feature')
        self.assertAlmostEqual(gain, 0.0481270304083)
        self.assertEqual(best_value, 1)

        data = pd.DataFrame([[1, 1], [1, 1], [1, 1], [1, 1], [1, 1], [1, 1], [0, 1], [0, 1], [0, 1], [1, 0], [1, 0], [0, 0], [0, 0], [0, 0]], columns=['feature', 'result'])
        gain, best_value = ML4T_Ex3_1.calc_info_gain(ML4T_Ex3_1.calc_variance, data, 'feature')
        self.assertAlmostEqual(gain, 0.015306122449)
        self.assertEqual(best_value, 1)

    def test_calc_variance(self):
        data = pd.DataFrame([[1, 1], [1, 1], [1, 1], [1, 1], [1, 1], [1, 1], [0, 1], [0, 1], [0, 1], [1, 0], [1, 0], [0, 0], [0, 0], [0, 0]], columns=['feature', 'result'])
        variance = ML4T_Ex3_1.calc_variance(data)
        self.assertAlmostEqual(variance, 0.22959183673469391)
