import unittest
from LinearRegression import *
import pandas

__author__ = 'bruce'


def get_answers():
    import pickle
    import os

    f = open(os.path.dirname(__file__)+"\\"+'testdata.txt', "rb")
    data = pickle.load(f, encoding='bytes')
    f.close()
    return data
    return ""

answers = get_answers()



class test_linear_regression(unittest.TestCase):

    def test_feature_normalize(self):
        # Load Data
        data = pandas.read_csv('ex1data2.txt', header=None).as_matrix()
        print(type(data))
        X = data[:, 0:2]
        y = data[:, 2]
        m = len(y)

        X, mu, sigma = featureNormalize(X)
        value = np.allclose(X, answers['X'])
        self.failUnless(np.allclose(X, answers['X']))
        self.failUnless(np.allclose(mu, answers['mu']))
        self.failUnless(np.allclose(sigma, answers['sigma']))
        return



if __name__ == '__main__': # pragma: no cover
    unittest.main()
