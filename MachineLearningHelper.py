from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import RFECV
from sklearn.tree import export_graphviz
import numpy as np

# noinspection PyPep8Naming
def get_best_features(X, y, logistic_regression = False, random_forest = False, decision_tree = False, cv=3, decision_tree_features=10, create_graph=False):

    orig_features = X.columns
    dict_of_bests = {}

    # What are the top features used by a decision tree?
    if decision_tree:
        tree = DecisionTreeClassifier(criterion='entropy', max_depth=len(X.columns)/2)
        tree.fit(X, y)

        # Find best features to train on
        importances = tree.feature_importances_
        indices = np.argsort(importances)[::-1]
        indices = indices[0:decision_tree_features]
        #std = np.std(tree.feature_importances_, axis=0)
        # Get 10 best features for Tree
        best_tree = []
        for f in indices:
            best_tree.append(X.columns[f])
        dict_of_bests['Decision Tree'] = best_tree

        if create_graph:
            feature_names = list(X.columns.values)
            export_graphviz(tree, out_file='TrainTree.dot', feature_names=feature_names)

    # Using Recursive feature elimination, what are the top features used by Logistic Regression?
    if logistic_regression:
        lr = LogisticRegression()
        # Create the RFE object and compute a cross-validated score.
        # The "accuracy" scoring is proportional to the number of correct
        rfecv = RFECV(estimator=lr, step=1, scoring='accuracy', cv=cv)
        rfecv.fit(X, y)
        ranks = rfecv.ranking_
        best_rec = rank_features(X, ranks)
        dict_of_bests['Logistic Regression'] = best_rec

    # Using Recursive feature elimination, what are the top features used by Random Forest?
    if random_forest:
        rfc = RandomForestClassifier(criterion='entropy', n_estimators=10, max_depth=5, max_features=0.07)
        rfecv = RFECV(estimator=rfc, step=1, scoring='accuracy', cv=cv)
        rfecv.fit(X, y)
        ranks = rfecv.ranking_
        best_rf = rank_features(X, ranks)
        dict_of_bests['Random Forest'] = best_rf

    list_of_bests = [[k, v] for k, v in dict_of_bests.items()]

    # noinspection PyPep8
    best_features = set.intersection(*map(set, [flist[1] for flist in list_of_bests]))
    #list_of_bests.append(['Intersection', best_features])
    dict_of_bests['Intersection'] = best_features

    for item in list_of_bests:
        print("")
        print("Best Features Using " + item[0] + ":")
        print(str(len(item[1])) + " out of " + str(len(orig_features)) + " features")
        print(item[1])

    return dict_of_bests


# noinspection PyPep8Naming
def rank_features(X, rankings):
    # The "accuracy" scoring is proportional to the number of correct classifications
    features = X.columns
    ranks = rankings
    best_features = []
    for i in range(0, len(features)):
        if ranks[i] == 1:
            best_features.append(features[i])
    return best_features


