import unittest
from LinearRegression import *
import pandas

__author__ = 'bruce'


def get_answers():
    import pickle
    import os

    f = open(os.path.dirname(__file__)+"\\"+'ex1testdata.txt', "rb")
    data = pickle.load(f, encoding='bytes')
    f.close()
    return data

answers = get_answers()



class test_linear_regression(unittest.TestCase):

    def test_feature_normalize(self):
        # Load Data
        data = pandas.read_csv('ex1data2.txt', header=None).as_matrix()
        print(type(data))
        X = data[:, 0:2]
        y = data[:, 2]
        m = len(y)

        X_norm, mu, sigma = featureNormalize(X)
        new_col = np.ones((np.size(X_norm, axis=0), 1), float)
        X_norm = np.append(new_col, X_norm, axis=1)
        self.failUnless(np.allclose(X_norm, answers['X']))
        self.failUnless(np.allclose(mu, answers['mu']))
        self.failUnless(np.allclose(sigma, answers['sigma']))

        theta = np.zeros((3, 1), float)
        theta, J1 = gradientDescentMulti(X_norm, y, theta, 0.1, 400)
        self.failUnless(np.allclose(theta, answers['theta_gd']))
        self.failUnless(np.allclose(J1, answers['J1']))
        price_gd = np.matmul(np.array([1, (1650-mu[0])/sigma[0], (3-mu[1])/sigma[1]]).reshape(1,3), theta.reshape(3,1)) # You should change this
        self.failUnless(np.allclose(price_gd, answers['price_gd']))

        X = np.column_stack((np.ones((m, 1)), X))
        theta = normalEqn(X, y)
        self.failUnless(np.allclose(theta, answers['theta_norm']))
        price_norm = np.matmul(np.array([[1, 1650, 3]]), theta)
        self.failUnless(np.allclose(price_norm, answers['price_norm']))

        self.assertAlmostEquals(price_gd[0,0], price_norm[0,0], places=2)

        cost = computeCostMulti(X_norm, y, answers['theta_gd'])
        self.assertAlmostEquals(cost, answers['cost'], places=2)

if __name__ == '__main__': # pragma: no cover
    unittest.main()
