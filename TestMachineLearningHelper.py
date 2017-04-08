from unittest import TestCase
import pandas as pd
import os
import MachineLearningHelper as mlh


def get_data(train_data):
    X_train = train_data.loc[:, ['Column1', 'Column2']]
    if 'Class' in train_data:
        y_train = train_data.loc[:,'Class']
    else:
        y_train = None

    return X_train, y_train



class TestMachineLearningHelper(TestCase):
    def test_get_best_recursive_features(self):
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

        best_features = mlh.get_best_recursive_features(X_train, y_train, logistic_regression=True, random_forest=True,
                                                    decision_tree=True)


        check = {'Logistic Regression': ['Column1', 'Column2'], 'Random Forest': ['Column1', 'Column2'], 'Intersection': {'Column2', 'Column1'}, 'Decision Tree': ['Column2', 'Column1']}
        self.assertEqual(best_features,check)
