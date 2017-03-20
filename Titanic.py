import pandas as pd
import numpy as np
import pickle
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
from importlib import reload
from sklearn.feature_selection import RFECV

from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import cross_val_score # Note: What is cross_val_predict?
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import GaussianNB



# Start log file
# logfile=os.getcwd()+'\\titanic.log'
# log.basicConfig(filename=logfile,level=log.INFO)
# log.info("Starting Titanic Training on " + str(time.strftime("%c")))


def munge_data(train_data, test_data=None):
    # X = pd.concat([train_data.ix[:,0:1], train_data.ix[:,2:3], train_data.ix[:,4:8], train_data.ix[:,9:]], axis=1) # .ix allows you to slice using labels and position, and concat pieces them back together. Not best way to do this.
    X_train = train_data[['Pclass', 'Sex', 'Age',  'SibSp', 'Parch', 'Fare', 'Embarked', 'Name', 'Cabin', 'Ticket']]
    if 'Survived' in train_data:
        y_train = train_data['Survived']
    else:
        y_train = None

    # Were we also passed the Test set for one hot encoding?
    if test_data is not None:
        X_test = test_data[['Pclass', 'Sex', 'Age',  'SibSp', 'Parch', 'Fare', 'Embarked', 'Name', 'Cabin', 'Ticket']]
        # Combine both sets for cleansing
        X_all = pd.concat([X_train, X_test])
    else:
        X_all = X_train

    # Out of name and cabin, create title and deck
    # For name we just want "Master", "Mr", "Miss", "Mrs" etc
    # For cabin we want just the deck letter
    X_all['Title'] = X_all.apply(lambda x: name_to_title(x['Name']), axis=1)
    X_all['Deck'] = X_all.apply(lambda x: cabin_to_deck(x['Cabin']), axis=1)
    # Impute missing ages based on title
    X_all['Adj Age'] = X_all.apply(lambda x: x['Age'] if ~np.isnan(x['Age']) else fix_age(x['Title']), axis=1)
    # Grab just the prefix for the ticket as I have no idea what to do with the rest of it
    X_all['TicketPre'] = get_ticket_prefix(X_all['Ticket'])
    # Set NA for any missing embarked
    X_all['Embarked'] = X_all.apply(lambda x: x['Embarked'] if (pd.isnull(x['Embarked']) == False) else "NA", axis=1)

    # Now Drop Name and Cabin because we no longer need them
    X_all = X_all.drop('Name', axis=1)
    X_all = X_all.drop('Cabin', axis=1)
    X_all = X_all.drop('Age', axis=1)
    X_all = X_all.drop('Ticket', axis=1)


    # Temp drops to try regression
    #X = X.drop('Title', axis=1)
    #X = X.drop('Embarked', axis=1)
    #X = X.drop('TicketPre', axis=1)
    #X = X.drop('Deck', axis=1)
    # Temp add new features for regression
    X_all['Age2'] = X_all['Adj Age']**2
    X_all['Fare2'] = X_all['Fare']**2


    """
    # Label Encoder way
    le = preprocessing.LabelEncoder()
    # Label Encode Sex
    le.fit(X.Sex.unique())
    X['Sex'] = le.transform(X['Sex'])
    # Label Encode Title
    le.fit(X.Title.unique())
    X['Title'] = le.transform(X['Title'])
    # Label Encode Deck
    le.fit(X.Deck.unique())
    X['Deck'] = le.transform(X['Deck'])
    # Label Encode Embarked
    le.fit(X.Embarked.unique())
    X['Embarked'] = le.transform(X['Embarked'])
    # Label Encode Ticket Prefix
    le.fit(X.TicketPre.unique())
    X['TicketPre'] = le.transform(X['TicketPre'])
    """

    # One Hot Encoding way
    cols_to_transform = ['Pclass', 'Sex', 'Embarked', 'Title', 'Deck', 'TicketPre']
    # First create columns by one hot encoding data and additional data (which will contain train and test data)
    X_all = pd.get_dummies(X_all, columns = cols_to_transform )
    #if not ('Embarked_NA' in X):
    #    X['Embarked_NA'] = 0
    #print(X)

    # Fix using Imputer -- fill in with mean
    imp = preprocessing.Imputer(missing_values='NaN', strategy='mean', axis=0)
    X_imputed = imp.fit_transform(X_all)
    X_all = pd.DataFrame(X_imputed, columns=X_all.columns)

    #X['PassengerId'] = X.apply(lambda x: int(x['PassengerId']), axis=1)
    # Old way
    #X = fix_missing_values(X, 'Adj Age')
    #X = fix_missing_values(X, 'Fare')

    # Test one field at a time
    #X = X.drop('Adj Age', axis=1)
    #X = X.drop('Pclass', axis=1)
    #X = X.drop('Fare', axis=1)  # 137
    #X = X.drop('SibSp', axis=1) #145
    #X = X.drop('Embarked', axis=1) #150

    # columns to drop with a decision tree
    #X = X.drop('TicketPre', axis=1) #150
    #X = X.drop('Deck', axis=1) #165
    #X = X.drop('Parch', axis=1) #165
    #X = X.drop('Title', axis=1) #168

    # Scale and center
    col_names = ['Adj Age', 'SibSp', 'Parch', 'Fare', 'Age2', 'Fare2']
    features = X_all[col_names]
    scaler = StandardScaler().fit(features.values)
    features = scaler.transform(features.values)
    X_all[col_names] = features
    #print(X)

    # Now split train and test apart again
    X_train = X_all[0:len(X_train)]
    X_test = X_all[len(X_train):len(X_all)]


    # Force order of columns
    #X = X[['Sex_female', 'Sex_male', 'Pclass_1', 'Pclass_2', 'Pclass_3', 'Adj Age', 'Age2', 'SibSp', 'Parch', 'Fare', 'Fare2', 'Embarked_C', 'Embarked_Q', 'Embarked_S', 'Embarked_NA']]

    return X_train, y_train, X_test



def get_ticket_prefix(tickets):
    regex = re.compile('[^a-zA-Z]')
    # First parameter is the replacement, second parameter is your input string
    # i.e. regex.sub('', 'ab3d*E')
    # Out: 'abdE'
    ticket_prefixes = [regex.sub('', ticket).upper() for ticket in tickets]
    ticket_prefixes = [ticket or 'NA' for ticket in ticket_prefixes]
    return ticket_prefixes




def fix_age(title):
    # Use average from training set for ages of titles
    if title == 'Capt': return 70.0
    elif title == 'Col': return 58.0
    elif title == 'Countess': return 33.0
    elif title == 'Dr': return 42.0
    elif title == 'Major': return 48.5
    elif title == 'Master': return 4.574167
    elif title == 'Miss': return 21.773973
    elif title == 'Mr': return 32.409774
    elif title == 'Mrs': return 35.900000
    elif title == 'Ms': return 28.000000
    elif title == 'NA': return 31.500000
    elif title == 'Rev': return 43.166667
    else: return -1.0


def name_to_title(name):
    name = name.upper()
    if 'MASTER'in name:
        return "Master"
    elif 'MISS'in name:
        return "Miss"
    elif "MR." in name or "MR " in name:
        return "Mr"
    elif "MRS." in name or "MRS " in name:
        return "Mrs"
    elif "MS." in name or "MS " in name:
        return "Ms"
    elif "REV." in name:
        return "Rev"
    elif "DR." in name:
        return "Dr"
    elif "CAPT." in name:
        return "Capt"
    elif "MAJOR" in name:
        return "Major"
    elif "COUNTESS" in name:
        return "Countess"
    elif "COL." in name or "COL " in name:
        return "Col"
    else:
        return "NA"

def cabin_to_deck(cabin):
    cabin = str(cabin)
    if cabin != 'nan':
        # Get first character as that is the deck
        return cabin[0]
    else:
        return "NA"




def get_best_features(X, y, feature_type="intersection", cv=3, decision_tree_features=10):
    orig_features = X.columns
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

    # classifications
    model = LogisticRegression()
    # Create the RFE object and compute a cross-validated score.
    # The "accuracy" scoring is proportional to the number of correct
    rfecv = RFECV(estimator=model, step=1, scoring='accuracy', cv=cv)
    rfecv.fit(X, y)
    ranks = rfecv.ranking_
    best_rec = rank_features(X, ranks)

    best_features = list(set(best_tree) & set(best_rec))

    print("Best Features Using Recursive Feature Elimination:")
    print(str(len(best_rec)) + " out of " + str(len(orig_features)) + " features")
    print(best_rec)
    print("Best Features Using Decision Tree (Top "+str(decision_tree_features)+"):")
    print(str(len(best_tree)) + " out of " + str(len(orig_features)) + " features")
    print(best_tree)
    print("Best Features Common to Both:")
    print(str(len(best_features)) + " out of " + str(len(orig_features)) + " features")
    print(best_features)


    if feature_type == "intersection":
        return best_features
    elif feature_type == "decision tree":
        return best_tree
    elif feature_type == "recursive":
        return best_rec
    else:
        return best_features, best_tree, best_rec
        #raise Exception("Bad Parameter for feature_type")



def rank_features(X, rankings):
    # The "accuracy" scoring is proportional to the number of correct classifications
    features = X.columns
    ranks = rankings
    best_features = []
    for i in range(0, len(features)):
        if ranks[i] == 1:
            best_features.append(features[i])
    return best_features


def train_ensemble_classifier(X, y, use_persisted_values=False, grid_search=False, weights = [1,1,1,1,1], cv=3, persist_name="bestparam"):
    # weights = [lr, svc, knn, rfc, nb]
    estimators = []
    orig_features = X.columns
    # Do we want to load past grid search parameters
    # Note: there is no easy way to save off recursive feature elimination with passing a list out, which I don't want to do
    if use_persisted_values == True:
        gs_best_params = load_best_parameters()

    if weights[0] != 0:
        # Logistic Regression
        lr = LogisticRegression()

        # Are we doing a grid search this time (for logistic regression)?
        if (grid_search == True):
            C_range = 10. ** np.arange(-2, 2)
            #penalty_options = ['l1', 'l2']
            param_grid = dict(C = C_range) #, lr__penalty = penalty_options)
            # Do grid search
            gs = GridSearchCV(estimator=lr, param_grid=param_grid, scoring='accuracy', cv=cv)
            # Do grid search
            lr = gs.fit(X, y)
            # Print out results of grid search
            print("LR: Best Cross Validation Score: " + str(gs.best_score_))
            print("LR: Best Parameters: " + str(gs.best_params_))
            # Save results of grid search for later use
            save_best_parameters(gs.best_params_, file_name=persist_name+"_lr_gs")

        estimators.append(('lr', lr))


    if weights[1] != 0:
        # Kernel SVC
        svc = SVC(probability=True, kernel='poly')

        # Are we doing a grid search this time (for SVC)?
        if (grid_search == True):
            # Grid search kernel SVCs
            C_range = 10. ** np.arange(-2, 2)
            #kernel_options = ['poly'] #['poly', 'rbf', 'sigmoid']
            param_grid = dict(C=C_range) #,svc__kernel = kernel_options)
            # Do grid search
            gs = GridSearchCV(estimator=svc, param_grid=param_grid, scoring='accuracy', cv=cv)
            # Do grid search
            lr = gs.fit(X, y)
            # Print out results of grid search
            print("SVC: Best Cross Validation Score: " + str(gs.best_score_))
            print("SVC: Best Parameters: " + str(gs.best_params_))
            # Save results of grid search for later use
            save_best_parameters(gs.best_params_, file_name=persist_name+"_svc_gs")

        estimators.append(('svc', svc))

    if(grid_search == True):
        print ("Parameter Grid for Grid Search: ")
        print(param_grid)

    if weights[2] != 0:
        # Create KNearestNeighbor model
        knn = KNeighborsClassifier(n_neighbors=4, p=2, metric='minkowski')
        estimators.append(('knn', knn))

    if weights[3] != 0:
        # Create RandomForest model
        rfc = RandomForestClassifier(criterion='entropy', n_estimators=1000, max_depth=len(X.columns)/2)
        estimators.append(('rfc', rfc))

    if weights[4] != 0:
        # Naive Bayes
        nb = GaussianNB()
        estimators.append(('nb', nb))

    # Adjust weights to remove 0s
    while 0 in weights: weights.remove(0)

    # Create majority vote ensemble classifier
    ensemble_clf = VotingClassifier(estimators=estimators, voting='soft', weights=weights)

    # Train final model
    ensemble_clf.fit(X, y)
    return ensemble_clf




def save_best_parameters(best_params, file_name='bestparams'):
    best_params = str(best_params)
    try:
        os.remove(os.path.dirname(__file__)+"\\"+file_name)
    finally:
        f = open(os.path.dirname(__file__)+"\\"+file_name, 'wb') # w for write, b for binary
    pickle.dump(best_params, f)
    f.close()


def load_best_parameters(file_name='bestparams'):
    f = open(os.path.dirname(__file__) + "\\" + file_name, "rb")
    data = pickle.load(f)
    f.close()
    return data



# Read in training data
trainfile = os.getcwd() + '\\Titanic\\train.csv'
train_data = pd.read_csv(trainfile)
# Grab Test data now so that we can build a proper one hot encoded data set
# Test data often has options not in training data, so we have to review both together
# If we want to one hot encoded things correctly
testfile = os.getcwd() + '\\Titanic\\test.csv'
test_data = pd.read_csv(testfile)

# Now munge the train data, but include test data so we get consistent one hot encoding
X_train, y_train, X_test = munge_data(train_data, test_data=test_data)

# Save out training data for bug fixing
X_train.to_csv(os.getcwd() + '\\Titanic\\CheckData.csv', index=False)
# Save out transformed Test Data for bug fixing
X_test.to_csv(os.getcwd() + '\\Titanic\\Xtest.csv')

"""
best_features = get_best_features(X_train, y_train, feature_type="recursive", cv=3)
# Use only best features
X_train = X_train[best_features]
X_test = X_test[best_features]
"""

clf = train_ensemble_classifier(X_train, y_train, weights = [1, 1, 1, 2, 0], grid_search=True, \
                                cv=10, persist_name="TitanicParams")

y_pred = clf.predict(X_train)
# returns statistics
print('Misclassified train samples: %d' % (y_train != y_pred).sum())
print('Accuracy of train set: %.2f' % accuracy_score(y_train, y_pred))

# Oops, cross validation has to run the whole thing multiple times!
# Try Kfold Cross Validation and get a more realistic score
#scores = cross_val_score(estimator=clf, X=X_train, y=y_train, cv=10)
#print('CV accuracy scores: %s' % scores)
#print('CV accuracy: #.3f +/- %.3f' % (np.mean(scores), np.std(scores)))

# Now predict using Test Data
y_pred = clf.predict(X_test)
y_pred = pd.DataFrame(y_pred)
y_pred.columns = ['Survived']

final_submission = pd.concat([test_data['PassengerId'], y_pred], axis=1)
final_submission.to_csv(os.getcwd() + '\\Titanic\\FinalSubmission.csv', index=False)

