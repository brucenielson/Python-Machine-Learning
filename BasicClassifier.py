import pandas as pd
import numpy as np
import os
import logging as log
import time
import sklearn
import math
import re
from PlotDecision import plot_decision_regions
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import export_graphviz
from sklearn.metrics import accuracy_score
from sklearn.feature_selection import RFECV
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.ensemble import VotingClassifier
# from importlib import reload

class BasicClassifier:
    def __init__(self):
        self.raw_data = None
        self.munge_data_function = None


    def __init__(self, train_file, test_file=None, munge_data_function=None):
        self.train_file_name = os.getcwd() + train_file
        self.raw_train_data = pd.read_csv(self.train_file_name)

        if test_file != None:
            self.test_file_name = os.getcwd() + test_file
            self.raw_test_data = pd.read_csv(self.train_file_name)
        else:
            self.test_file_name = None

        self.munge_data_function = munge_data_function
        self.train_X, self.train_y = self.munge_data_function(self.raw_train_data)
        self.test_X, self.test_y = self.munge_data_function(self.raw_test_data)



    def train_classifier(X, y):
        # Create train / test split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

        # Create comprehensive pipeline of:
        # Decision Tree, Random Forest, LogisticRegression, KNN, SVC
        dtc = DecisionTreeClassifier(criterion='entropy')
        rfc = RandomForestClassifier(criterion='entropy', n_estimators=100, max_depth=len(X.columns))
        knn = KNeighborsClassifier(n_neighbors=4, p=2, metric='minkowski')
        lr = LogisticRegression(max_iter=1000)
        svc = SVC(max_iter=1000)
        ensemble_clf = VotingClassifier(estimators=[('dtc', dtc), ('rfc', rfc), ('knn', knn), ('lr', lr)], voting='soft')

        # Create pipelines as needed for scaling data

        # Grid Search p. 185-186 in Python Machine Learning
        #param_range = [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]
        #param_grid = [{'C': param_range}]

        # LogisticRegression using One Hot Encoded Data

        # Try recursive feature elimination first
        # Create the RFE object and compute a cross-validated score.
        lr_model = LogisticRegression(max_iter=1000, C=0.01)

        # The "accuracy" scoring is proportional to the number of correct classifications
        rfecv = RFECV(estimator=ensemble_clf, step=1, scoring='accuracy')
        rfecv.fit(X, y)
        features = X.columns
        ranks = rfecv.ranking_
        best_features = []
        for i in range(0, len(features)):
            if ranks[i] == 1:
                best_features.append(features[i])
        X_best_features = X[best_features]
        print("Best Features: ")
        print(best_features)

        # Using best features, fit the model
        lr_model.fit(X, y)

        return lr_model, best_features

