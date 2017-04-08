from unittest import TestCase
import pandas as pd
import os
import EnsembleClassifier as classifier
from sklearn.metrics import accuracy_score


def get_data(train_data):
    X_train = train_data.loc[:, ['Column1', 'Column2']]
    if 'Class' in train_data:
        y_train = train_data.loc[:,'Class']
    else:
        y_train = None

    return X_train, y_train



class TestTrain_ensemble_classifier(TestCase):
    def test_train_ensemble_classifier(self):
        # Read in training data
        trainfile = os.getcwd() + '\\Data\\UnitTestTrain.csv'
        train_data = pd.read_csv(trainfile)
        # Grab Test data now so that we can build a proper one hot encoded data set
        # Test data often has options not in training data, so we have to review both together
        # If we want to one hot encoded things correctly
        testfile = os.getcwd() + '\\Data\\UnitTestTest.csv'
        test_data = pd.read_csv(testfile)

        # Now munge the train data, but include test data so we get consistent one hot encoding
        X_train, y_train = get_data(train_data)
        X_test, y_test = get_data(test_data)

        # weights = [lr, svc, knn, rfc, nb]
        clf = classifier.train_ensemble_classifier(X_train, y_train, weights=[1, 1, 1, 1, 1], grid_search=False,
                                                   persist_name="UnitTestParams", use_persisted_values=False)

        y_pred = clf.predict(X_train)

        # returns statistics
        wrong1 = (y_train != y_pred).sum()
        accuracy1 = accuracy_score(y_train, y_pred)
        self.assertTrue(wrong1 == 0 or wrong1 == 1)
        self.assertTrue(accuracy1 >=  0.95 and accuracy1 <=1.0)

        # Now predict using Test Data
        y_pred = clf.predict(X_test)

        # Compare prediction to actuals
        self.assertEqual(len(y_pred), len(y_test))
        wrong2 = (y_test != y_pred).sum()
        accuracy2 = accuracy_score(y_test, y_pred)
        self.assertTrue(wrong2  == 5 or wrong2 == 6)
        self.assertTrue(accuracy2 >= 0.7 and accuracy2 <= 0.75)

        # weights = [lr, svc, knn, rfc, nb]
        clf = classifier.train_ensemble_classifier(X_train, y_train, weights=[1, 1, 1, 1, 1], grid_search=False,
                                                   persist_name="UnitTestParams", use_persisted_values=True)

        y_pred = clf.predict(X_train)

        # returns statistics
        wrong1 = (y_train != y_pred).sum()
        accuracy1 = accuracy_score(y_train, y_pred)
        self.assertTrue(wrong1 == 0 or wrong1 == 1)
        self.assertTrue(accuracy1 >=  0.95 and accuracy1 <=1.0)

        # Now predict using Test Data
        y_pred = clf.predict(X_test)

        # Compare prediction to actuals
        self.assertEqual(len(y_pred), len(y_test))
        wrong2 = (y_test != y_pred).sum()
        accuracy2 = accuracy_score(y_test, y_pred)
        self.assertTrue(wrong2  == 5 or wrong2 == 6)
        self.assertTrue(accuracy2 >= 0.7 and accuracy2 <= 0.75)