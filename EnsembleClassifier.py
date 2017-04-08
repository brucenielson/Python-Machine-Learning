import numpy as np
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import VotingClassifier
import pickle
from sklearn.naive_bayes import GaussianNB
# from importlib import reload

# noinspection PyPep8Naming
def do_grid_search(X, y, estimator, parameters, persist_name="bestparams", cv=3):
    # Do grid search
    est = estimator()
    gs = GridSearchCV(estimator=est, param_grid=parameters, scoring='accuracy', cv=cv)
    # Do grid search
    gs.fit(X, y)
    est = estimator(**gs.best_params_)
    # Print out results of grid search
    print("")
    print("Parameter Grid for Grid Search on " + str(type(est).__name__))
    print(parameters)
    print("")
    print("Grid Search Results:")
    print("Best Cross Validation Score: " + str(gs.best_score_))
    print("Best Parameters: " + str(gs.best_params_))
    # Save results of grid search for later use
    save_best_parameters(gs.best_params_, file_name=persist_name)
    return est


# noinspection PyPep8Naming
def create_classifier(X, y, clf, name, grid_search, param_grid, use_persisted_values, persist_name, cv=3):
    persist_name = persist_name + "_"+name+"_gs"
    # Are we doing a grid search this time (for logistic regression)?
    if grid_search:
        clf = do_grid_search(X, y, clf, param_grid, persist_name, cv=cv)
    elif use_persisted_values:
        params = load_best_parameters(persist_name)
        if 'n_estimators' in params:
            params['n_estimators'] = params['n_estimators'] * 10
        clf = clf(**params)
        # Print out results of loaded parrameters
        print("")
        print("Saved Parameters for " + str(type(clf).__name__) + ": ")
        print(params)
    else:
        if name.lower() == 'svc':
            clf = clf(probability=True)
        else:
            clf = clf()

    return clf


# noinspection PyPep8Naming,PyTypeChecker,PyTypeChecker
def train_ensemble_classifier(X, y, use_persisted_values=False, grid_search=False, weights=(1, 1, 1, 1, 1), cv=3, persist_name="bestparams"):
    # weights = [lr, svc, knn, rfc, nb]
    estimators = []
    # Do we want to load past grid search parameters
    # Note: there is no easy way to save off recursive feature elimination with passing a list out, which I don't want to do

    # Logistic Regression
    if weights[0] != 0:
        # Create parameters
        # noinspection PyPep8Naming
        C_range = 10. ** np.arange(-4, 4)
        penalty_options = ['l1', 'l2']
        param_grid = dict(C=C_range, penalty=penalty_options)
        # Create classifier
        clf = create_classifier(X, y, LogisticRegression, 'lr', grid_search, param_grid, use_persisted_values, persist_name, cv=cv)
        # Add to list of estimators
        estimators.append(('lr', clf))

    # Kernel SVC
    if weights[1] != 0:
        # Create parameters
        # Grid search kernel SVCs
        # noinspection PyPep8Naming
        C_range = 10. ** np.arange(-4, 4)
        kernel_options = ['poly', 'rbf', 'sigmoid']
        param_grid = dict(C=C_range, kernel=kernel_options, probability=[True])
        # Create classifier
        clf = create_classifier(X, y, SVC, 'svc', grid_search, param_grid, use_persisted_values, persist_name, cv=cv)
        # Add to list of estimators
        estimators.append(('svc', clf))

    # KNN
    if weights[2] != 0:
        # Create parameters
        # Grid search KNN
        n_neighbors = range(3, 9)
        algorithm = ['ball_tree', 'kd_tree', 'brute']
        p = [1, 2]
        metric = ['euclidean', 'minkowski', 'manhattan']
        weight = ['uniform', 'distance']
        param_grid = dict(n_neighbors=n_neighbors, p=p, metric=metric, weights=weight, algorithm=algorithm)
        # Create classifier
        clf = create_classifier(X, y, KNeighborsClassifier, 'knn', grid_search, param_grid, use_persisted_values, persist_name, cv=cv)
        # Add to list of estimators
        estimators.append(('knn', clf))

    # Random Forest
    if weights[3] != 0:
        # Create parameters
        # Grid search Random Forest Classifier
        num_features = len(X.columns)
        max_features = ['sqrt', 'log2', 0.01, 0.10]
        if num_features >= 5:
            max_features.append(5)
        if num_features >= 25:
            max_features.append(25)
        max_depth = [3, 5, 9, 15, 25]
        criterion = ['gini', 'entropy']
        min_samples_split = [2, 4, 6]
        min_samples_leaf = [1, 3, 5, 0.01, 0.1]
        n_estimators = [100]
        param_grid = dict(max_features=max_features, max_depth=max_depth, criterion=criterion, min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf, n_estimators=n_estimators)
        # Create classifier
        clf = create_classifier(X, y, RandomForestClassifier, 'rfc', grid_search, param_grid, use_persisted_values, persist_name, cv=cv)
        # Add to list of estimators
        estimators.append(('rfc', clf))

    # Naive Bayes
    if weights[4] != 0:
        # Naive Bayes
        nb = GaussianNB()
        estimators.append(('nb', nb))

    # Adjust weights to remove 0s
    while 0 in weights: weights.remove(0)
    print("")
    print('Estimators: ' + str([item[0] for item in estimators]))
    # Create majority vote ensemble classifier
    ensemble_clf = VotingClassifier(estimators=estimators, voting='soft', weights=weights)

    # Train final model
    ensemble_clf.fit(X, y)
    return ensemble_clf



def save_best_parameters(best_params, file_name='bestparams'):
    # noinspection PyBroadException
    try:
        os.remove(os.path.dirname(__file__)+"\\Persistence\\"+file_name)
    except:
        pass
    finally:
        f = open(os.path.dirname(__file__)+"\\Persistence\\"+file_name, 'wb') # w for write, b for binary
    pickle.dump(best_params, f)
    f.close()



def load_best_parameters(file_name='bestparams'):
    # noinspection PyBroadException
    try:
        f = open(os.path.dirname(__file__) + "\\Persistence\\" + file_name, "rb")
        data = pickle.load(f)
        f.close()
    except:
        data = {}
    return data
