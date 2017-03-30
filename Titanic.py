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
    X_all['Cabin'] = X_all.apply(lambda x: x['Cabin'] if (pd.isnull(x['Cabin']) == False) else "NA", axis=1)
    X_all['Ticket'] = X_all.apply(lambda x: x['Ticket'] if (pd.isnull(x['Ticket']) == False) else "NA", axis=1)

    # Now Drop Name and Cabin because we no longer need them
    X_all = X_all.drop('Name', axis=1)
    #X_all = X_all.drop('Cabin', axis=1)
    X_all = X_all.drop('Age', axis=1)
    #X_all = X_all.drop('Ticket', axis=1)


    # Temp drops to try regression
    #X = X.drop('Title', axis=1)
    #X = X.drop('Embarked', axis=1)
    #X = X.drop('TicketPre', axis=1)
    #X = X.drop('Deck', axis=1)


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
    cols_to_transform = ['Pclass', 'Sex', 'Embarked', 'Title', 'Deck', 'TicketPre', 'Cabin', 'Ticket']
    # First create columns by one hot encoding data and additional data (which will contain train and test data)
    X_all = pd.get_dummies(X_all, columns = cols_to_transform )
    #if not ('Embarked_NA' in X):
    #    X['Embarked_NA'] = 0
    #print(X)

    # Fix using Imputer -- fill in with mean for columns with continuous values
    imp = preprocessing.Imputer(missing_values='NaN', strategy='mean', axis=0)
    X_all[['Fare', 'Adj Age']] = imp.fit_transform(X_all[['Fare', 'Adj Age']])
    #X_all = pd.DataFrame(X_imputed, columns=X_all.columns)

    # Temp add new features for regression
    X_all['Age2'] = X_all['Adj Age']**2
    X_all['Fare2'] = X_all['Fare']**2


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
    col_names = ['Adj Age', 'SibSp', 'Parch', 'Age2', 'Fare', 'Fare2' ]
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

    print("# of Columns:")
    print(len(X_train.columns))
    #print(X_train)

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




def get_best_features(X, y, logistic_regression = False, random_forest = False, decision_tree = False, cv=3, decision_tree_features=10):

    orig_features = X.columns
    dict_of_bests = {}

    # What are the top features used by a decision tree?
    if decision_tree == True:
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

    # Using Recursive feature elimination, what are the top features used by Logistic Regression?
    if logistic_regression == True:
        lr = LogisticRegression()
        # Create the RFE object and compute a cross-validated score.
        # The "accuracy" scoring is proportional to the number of correct
        rfecv = RFECV(estimator=lr, step=1, scoring='accuracy', cv=cv)
        rfecv.fit(X, y)
        ranks = rfecv.ranking_
        best_rec = rank_features(X, ranks)
        dict_of_bests['Logistic Regression'] = best_rec

    # Using Recursive feature elimination, what are the top features used by Random Forest?
    if random_forest == True:
        rfc = RandomForestClassifier(criterion='entropy', n_estimators=10, max_depth=5, max_features=0.07)
        rfecv = RFECV(estimator=rfc, step=1, scoring='accuracy', cv=cv)
        rfecv.fit(X, y)
        ranks = rfecv.ranking_
        best_rf = rank_features(X, ranks)
        dict_of_bests['Random Forest'] =  best_rf

    list_of_bests = [[k, v] for k, v in dict_of_bests.items()]

    best_features = set.intersection(*map(set,[flist[1] for flist in list_of_bests]))
    list_of_bests.append(('Intersection', best_features))

    for item in list_of_bests:
        print("")
        print("Best Features Using " + item[0] + ":")
        print(str(len(item[1])) + " out of " + str(len(orig_features)) + " features")
        print(item[1])

    return dict_of_bests


def rank_features(X, rankings):
    # The "accuracy" scoring is proportional to the number of correct classifications
    features = X.columns
    ranks = rankings
    best_features = []
    for i in range(0, len(features)):
        if ranks[i] == 1:
            best_features.append(features[i])
    return best_features



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


def create_classifier(X, y, clf, name, grid_search, param_grid, use_persisted_values, persist_name):
    persist_name = persist_name + "_"+name+"_gs"
    # Are we doing a grid search this time (for logistic regression)?
    if (grid_search == True):
        clf = do_grid_search(X, y, clf, param_grid, persist_name)
    elif use_persisted_values == True:
        params = load_best_parameters(persist_name)
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



def train_ensemble_classifier(X, y, use_persisted_values=False, grid_search=False, weights = [1,1,1,1,1], cv=3, persist_name="bestparams"):
    # weights = [lr, svc, knn, rfc, nb]
    estimators = []
    orig_features = X.columns
    # Do we want to load past grid search parameters
    # Note: there is no easy way to save off recursive feature elimination with passing a list out, which I don't want to do

    # Logistic Regression
    if weights[0] != 0:
        # Create parameters
        C_range = 10. ** np.arange(-4, 4)
        penalty_options = ['l1', 'l2']
        param_grid = dict(C=C_range, penalty=penalty_options)
        # Create classifier
        clf = create_classifier(X, y, LogisticRegression, 'lr', grid_search, param_grid, use_persisted_values, persist_name)
        # Add to list of estimators
        estimators.append(('lr', clf))

    # Kernel SVC
    if weights[1] != 0:
        # Create parameters
        # Grid search kernel SVCs
        C_range = 10. ** np.arange(-4, 4)
        kernel_options = ['poly', 'rbf', 'sigmoid']
        param_grid = dict(C=C_range, kernel=kernel_options, probability=[True])
        # Create classifier
        clf = create_classifier(X, y, SVC, 'svc', grid_search, param_grid, use_persisted_values, persist_name)
        # Add to list of estimators
        estimators.append(('svc', clf))

    # KNN
    if weights[2] != 0:
        # Create parameters
        # Grid search kernel SVCs
        n_neighbors = np.arange(3, 9)
        #algorithm = ['ball_tree', 'kd_tree', 'brute']
        p = [1,2]
        metric = ['euclidean', 'minkowski'] #, 'manhattan']
        #weight = ['uniform', 'distance']
        param_grid = dict(n_neighbors=n_neighbors, p=p, metric=metric) #, weights=weight, algorithm=algorithm)
        # Create classifier
        clf = create_classifier(X, y, KNeighborsClassifier, 'knn', grid_search, param_grid, use_persisted_values, persist_name)
        # Add to list of estimators
        estimators.append(('knn', clf))


    if weights[3] != 0:
        # Create RandomForest model
        rfc = RandomForestClassifier(criterion='entropy', n_estimators=1000, max_depth=5, max_features=0.1) # len(X.columns)/2)
        estimators.append(('rfc', rfc))

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
    try:
        os.remove(os.path.dirname(__file__)+"\\"+file_name)
    except:
        pass
    finally:
        f = open(os.path.dirname(__file__)+"\\"+file_name, 'wb') # w for write, b for binary
    pickle.dump(best_params, f)
    f.close()


def load_best_parameters(file_name='bestparams'):
    try:
        f = open(os.path.dirname(__file__) + "\\" + file_name, "rb")
        data = pickle.load(f)
        f.close()
    except:
        data = {}
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

#X_train = X_train[['Cabin_E58', 'TicketPre_C', 'Ticket_7552', 'Ticket_345764', 'Ticket_364849', 'Cabin_E44', 'Ticket_A/5. 10482', 'Cabin_D33', 'Ticket_347071', 'Ticket_29105', 'Ticket_382649', 'Ticket_PC 17611', 'Ticket_315098', 'Ticket_C 17369', 'Cabin_A10', 'Ticket_113773', 'Ticket_2678', 'Cabin_C95', 'Ticket_13049', 'Ticket_11967', 'TicketPre_STONO', 'Ticket_345779', 'Ticket_PC 17473', 'Ticket_36967', 'Ticket_347470', 'Cabin_C104', 'Ticket_2663', 'Ticket_29106', 'Cabin_C52', 'Ticket_2699', 'Ticket_386525', 'Ticket_113804', 'Ticket_244278', 'Ticket_SOTON/O.Q. 392078', 'Ticket_PC 17483', 'Ticket_A/5. 3337', 'Ticket_3101278', 'Ticket_A/5. 3336', 'Cabin_C124', 'Ticket_113781', 'Ticket_113503', 'Ticket_7546', 'Ticket_STON/O 2. 3101285', 'Ticket_W./C. 6609', 'Cabin_E63', 'Ticket_315084', 'Ticket_11753', 'Deck_NA', 'Ticket_349909', 'Cabin_C106', 'Ticket_A/5 3540', 'Ticket_111426', 'Ticket_19943', 'Ticket_2627', 'Ticket_343095', 'Ticket_250651', 'Ticket_347080', 'Ticket_SW/PP 751', 'Ticket_347077', 'Ticket_13213', 'Ticket_244270', 'Ticket_W./C. 6608', 'Ticket_4138', 'Ticket_PC 17485', 'Ticket_110465', 'Ticket_PC 17558', 'Cabin_C47', 'TicketPre_WEP', 'Cabin_A19', 'Ticket_248698', 'Ticket_7553', 'Cabin_C111', 'Ticket_STON/O2. 3101271', 'Cabin_NA', 'Ticket_237798', 'Ticket_PC 17474', 'Deck_C', 'Cabin_C148', 'Title_Mr', 'Cabin_C49', 'Ticket_111428', 'Ticket_347083', 'Ticket_347085', 'Ticket_SOTON/O.Q. 392087', 'Deck_D', 'Cabin_D6', 'Ticket_237671', 'Ticket_113788', 'Ticket_248747', 'Cabin_A31', 'Ticket_367228', 'Cabin_D48', 'TicketPre_WC', 'Ticket_CA 2144', 'Ticket_347088', 'Ticket_330935', 'Cabin_D45', 'Cabin_E8', 'Ticket_2691', 'Ticket_19947', 'Ticket_113786', 'Ticket_13567', 'Cabin_B86', 'Ticket_113051', 'Ticket_19988', 'Ticket_W./C. 14258', 'Ticket_364846', 'Ticket_113760', 'Ticket_364850', 'Ticket_STON/O 2. 3101286', 'Ticket_382651', 'Cabin_D19', 'Cabin_B20', 'Ticket_S.O./P.P. 3', 'Ticket_11668', 'Ticket_113055', 'Ticket_SOTON/OQ 392089', 'TicketPre_SOPP', 'Deck_G', 'Cabin_C92', 'Ticket_16988', 'Ticket_349240', 'Title_Master', 'Ticket_11774', 'Ticket_2668', 'Ticket_363291', 'Ticket_SC/PARIS 2146', 'Ticket_4135', 'Ticket_STON/O2. 3101290', 'Ticket_STON/O 2. 3101289', 'Ticket_312991', 'Title_NA', 'Ticket_111320', 'Ticket_PC 17595', 'Ticket_SC/Paris 2163', 'Cabin_C22 C26', 'Cabin_A20', 'Pclass_1', 'Ticket_4134', 'Ticket_5727', 'TicketPre_PP', 'Sex_male', 'Ticket_PP 9549', 'Cabin_D26', 'Ticket_2908', 'Ticket_3101281', 'Ticket_365226', 'Ticket_2665', 'Ticket_343120', 'Ticket_S.O.C. 14879', 'Ticket_PC 17476', 'Ticket_113806', 'Cabin_B41', 'Ticket_3101295', 'Ticket_CA. 2314', 'Ticket_17453', 'Ticket_PC 17593', 'Cabin_A6', 'Ticket_347081', 'Ticket_345773', 'Ticket_11751', 'Ticket_17463', 'Ticket_350034', 'Ticket_2651', 'Embarked_S', 'Pclass_2', 'Title_Rev', 'Cabin_A26', 'Cabin_D30', 'Ticket_349237', 'Ticket_347742', 'Ticket_2620', 'Ticket_113501', 'Ticket_364516', 'Ticket_364848', 'Ticket_19952', 'Ticket_111427', 'Cabin_C126', 'Ticket_367230', 'Ticket_345763', 'Cabin_E46', 'Ticket_239865', 'Ticket_110564', 'TicketPre_LINE', 'Ticket_112379', 'Ticket_1601', 'Ticket_27042', 'Cabin_B58 B60', 'Ticket_4133', 'Title_Mrs', 'Ticket_350417', 'Ticket_19996', 'Ticket_347054', 'Ticket_345774', 'Ticket_330909', 'Ticket_2677', 'Cabin_E25', 'Ticket_C.A. 2315', 'Ticket_367226', 'TicketPre_FCC', 'Ticket_35281', 'Cabin_C82', 'Sex_female', 'Ticket_C.A. 2673', 'Cabin_C93', 'Ticket_4136', 'Ticket_2653', 'Ticket_347073', 'Ticket_LINE', 'Ticket_345572', 'Cabin_G6', 'Ticket_382652', 'Ticket_315096', 'Ticket_PC 17758', 'Ticket_349245', 'Ticket_2661', 'Ticket_239853', 'Cabin_E10', 'Cabin_C70', 'Cabin_D35', 'Ticket_C.A. 37671', 'Ticket_347082', 'Cabin_D', 'Ticket_347087', 'Cabin_E17', 'Ticket_244252', 'Cabin_A23', 'Cabin_A32', 'Ticket_347089', 'Ticket_PC 17572', 'Ticket_370129', 'SibSp', 'Ticket_112277', 'Ticket_4137', 'Ticket_S.W./PP 752', 'Cabin_E24', 'Ticket_695', 'Ticket_350407', 'Pclass_3', 'Ticket_244373', 'Cabin_D56', 'Ticket_350406', 'Ticket_17474', 'Ticket_349236', 'Ticket_113794', 'Ticket_PC 17475', 'Ticket_220845', 'Ticket_350046', 'Ticket_2666', 'Adj Age', 'Cabin_B49', 'Ticket_350043', 'Ticket_7598', 'Ticket_STON/O 2. 3101269', 'Ticket_65306', 'TicketPre_SWPP', 'Ticket_W.E.P. 5734', 'Ticket_113050', 'Cabin_B102', 'Cabin_B38', 'Deck_E', 'Cabin_B96 B98', 'Ticket_111369', 'Ticket_2689', 'Cabin_E12', 'Ticket_113767', 'Ticket_STON/O 2. 3101288', 'Ticket_3101265', 'Cabin_D46', 'Cabin_E77']]
#X_test = X_test[['Cabin_E58', 'TicketPre_C', 'Ticket_7552', 'Ticket_345764', 'Ticket_364849', 'Cabin_E44', 'Ticket_A/5. 10482', 'Cabin_D33', 'Ticket_347071', 'Ticket_29105', 'Ticket_382649', 'Ticket_PC 17611', 'Ticket_315098', 'Ticket_C 17369', 'Cabin_A10', 'Ticket_113773', 'Ticket_2678', 'Cabin_C95', 'Ticket_13049', 'Ticket_11967', 'TicketPre_STONO', 'Ticket_345779', 'Ticket_PC 17473', 'Ticket_36967', 'Ticket_347470', 'Cabin_C104', 'Ticket_2663', 'Ticket_29106', 'Cabin_C52', 'Ticket_2699', 'Ticket_386525', 'Ticket_113804', 'Ticket_244278', 'Ticket_SOTON/O.Q. 392078', 'Ticket_PC 17483', 'Ticket_A/5. 3337', 'Ticket_3101278', 'Ticket_A/5. 3336', 'Cabin_C124', 'Ticket_113781', 'Ticket_113503', 'Ticket_7546', 'Ticket_STON/O 2. 3101285', 'Ticket_W./C. 6609', 'Cabin_E63', 'Ticket_315084', 'Ticket_11753', 'Deck_NA', 'Ticket_349909', 'Cabin_C106', 'Ticket_A/5 3540', 'Ticket_111426', 'Ticket_19943', 'Ticket_2627', 'Ticket_343095', 'Ticket_250651', 'Ticket_347080', 'Ticket_SW/PP 751', 'Ticket_347077', 'Ticket_13213', 'Ticket_244270', 'Ticket_W./C. 6608', 'Ticket_4138', 'Ticket_PC 17485', 'Ticket_110465', 'Ticket_PC 17558', 'Cabin_C47', 'TicketPre_WEP', 'Cabin_A19', 'Ticket_248698', 'Ticket_7553', 'Cabin_C111', 'Ticket_STON/O2. 3101271', 'Cabin_NA', 'Ticket_237798', 'Ticket_PC 17474', 'Deck_C', 'Cabin_C148', 'Title_Mr', 'Cabin_C49', 'Ticket_111428', 'Ticket_347083', 'Ticket_347085', 'Ticket_SOTON/O.Q. 392087', 'Deck_D', 'Cabin_D6', 'Ticket_237671', 'Ticket_113788', 'Ticket_248747', 'Cabin_A31', 'Ticket_367228', 'Cabin_D48', 'TicketPre_WC', 'Ticket_CA 2144', 'Ticket_347088', 'Ticket_330935', 'Cabin_D45', 'Cabin_E8', 'Ticket_2691', 'Ticket_19947', 'Ticket_113786', 'Ticket_13567', 'Cabin_B86', 'Ticket_113051', 'Ticket_19988', 'Ticket_W./C. 14258', 'Ticket_364846', 'Ticket_113760', 'Ticket_364850', 'Ticket_STON/O 2. 3101286', 'Ticket_382651', 'Cabin_D19', 'Cabin_B20', 'Ticket_S.O./P.P. 3', 'Ticket_11668', 'Ticket_113055', 'Ticket_SOTON/OQ 392089', 'TicketPre_SOPP', 'Deck_G', 'Cabin_C92', 'Ticket_16988', 'Ticket_349240', 'Title_Master', 'Ticket_11774', 'Ticket_2668', 'Ticket_363291', 'Ticket_SC/PARIS 2146', 'Ticket_4135', 'Ticket_STON/O2. 3101290', 'Ticket_STON/O 2. 3101289', 'Ticket_312991', 'Title_NA', 'Ticket_111320', 'Ticket_PC 17595', 'Ticket_SC/Paris 2163', 'Cabin_C22 C26', 'Cabin_A20', 'Pclass_1', 'Ticket_4134', 'Ticket_5727', 'TicketPre_PP', 'Sex_male', 'Ticket_PP 9549', 'Cabin_D26', 'Ticket_2908', 'Ticket_3101281', 'Ticket_365226', 'Ticket_2665', 'Ticket_343120', 'Ticket_S.O.C. 14879', 'Ticket_PC 17476', 'Ticket_113806', 'Cabin_B41', 'Ticket_3101295', 'Ticket_CA. 2314', 'Ticket_17453', 'Ticket_PC 17593', 'Cabin_A6', 'Ticket_347081', 'Ticket_345773', 'Ticket_11751', 'Ticket_17463', 'Ticket_350034', 'Ticket_2651', 'Embarked_S', 'Pclass_2', 'Title_Rev', 'Cabin_A26', 'Cabin_D30', 'Ticket_349237', 'Ticket_347742', 'Ticket_2620', 'Ticket_113501', 'Ticket_364516', 'Ticket_364848', 'Ticket_19952', 'Ticket_111427', 'Cabin_C126', 'Ticket_367230', 'Ticket_345763', 'Cabin_E46', 'Ticket_239865', 'Ticket_110564', 'TicketPre_LINE', 'Ticket_112379', 'Ticket_1601', 'Ticket_27042', 'Cabin_B58 B60', 'Ticket_4133', 'Title_Mrs', 'Ticket_350417', 'Ticket_19996', 'Ticket_347054', 'Ticket_345774', 'Ticket_330909', 'Ticket_2677', 'Cabin_E25', 'Ticket_C.A. 2315', 'Ticket_367226', 'TicketPre_FCC', 'Ticket_35281', 'Cabin_C82', 'Sex_female', 'Ticket_C.A. 2673', 'Cabin_C93', 'Ticket_4136', 'Ticket_2653', 'Ticket_347073', 'Ticket_LINE', 'Ticket_345572', 'Cabin_G6', 'Ticket_382652', 'Ticket_315096', 'Ticket_PC 17758', 'Ticket_349245', 'Ticket_2661', 'Ticket_239853', 'Cabin_E10', 'Cabin_C70', 'Cabin_D35', 'Ticket_C.A. 37671', 'Ticket_347082', 'Cabin_D', 'Ticket_347087', 'Cabin_E17', 'Ticket_244252', 'Cabin_A23', 'Cabin_A32', 'Ticket_347089', 'Ticket_PC 17572', 'Ticket_370129', 'SibSp', 'Ticket_112277', 'Ticket_4137', 'Ticket_S.W./PP 752', 'Cabin_E24', 'Ticket_695', 'Ticket_350407', 'Pclass_3', 'Ticket_244373', 'Cabin_D56', 'Ticket_350406', 'Ticket_17474', 'Ticket_349236', 'Ticket_113794', 'Ticket_PC 17475', 'Ticket_220845', 'Ticket_350046', 'Ticket_2666', 'Adj Age', 'Cabin_B49', 'Ticket_350043', 'Ticket_7598', 'Ticket_STON/O 2. 3101269', 'Ticket_65306', 'TicketPre_SWPP', 'Ticket_W.E.P. 5734', 'Ticket_113050', 'Cabin_B102', 'Cabin_B38', 'Deck_E', 'Cabin_B96 B98', 'Ticket_111369', 'Ticket_2689', 'Cabin_E12', 'Ticket_113767', 'Ticket_STON/O 2. 3101288', 'Ticket_3101265', 'Cabin_D46', 'Cabin_E77']]

# Save out training data for bug fixing
X_train.to_csv(os.getcwd() + '\\Titanic\\CheckData.csv', index=False)
# Save out transformed Test Data for bug fixing
X_test.to_csv(os.getcwd() + '\\Titanic\\Xtest.csv')

"""
best_features = get_best_features(X_train, y_train, logistic_regression=True, cv=100)
# Use only best features
X_train = X_train[best_features['Logistic Regression']]
X_test = X_test[best_features['Logistic Regression']]
"""
#X_train = X_train[['SibSp', 'Adj Age', 'Pclass_3', 'Sex_female', 'Pclass_3']]
#X_test = X_test[['SibSp', 'Adj Age', 'Pclass_3', 'Sex_female', 'Pclass_3']]


# weights = [lr, svc, knn, rfc, nb]
clf = train_ensemble_classifier(X_train, y_train, weights = [0, 0, 1, 0, 0], grid_search=True, \
                                cv=10, persist_name="TitanicParams", use_persisted_values=True)

y_pred = clf.predict(X_train)
# returns statistics
print("")
print("Results of Predict:")
print('Misclassified train samples: %d' % (y_train != y_pred).sum())
print('Accuracy of train set: %.2f' % accuracy_score(y_train, y_pred))

"""
# Oops, cross validation has to run the whole thing multiple times!
# Try Kfold Cross Validation and get a more realistic score
scores = cross_val_score(estimator=clf, X=X_train, y=y_train, cv=10)
print("")
print("Results of Cross Validation:")
#print('CV accuracy scores: %s' % scores)
print('CV accuracy: %.3f +/- %.3f' % (np.mean(scores), np.std(scores)))
"""

# Now predict using Test Data
y_pred = clf.predict(X_test)
y_pred = pd.DataFrame(y_pred)
y_pred.columns = ['Survived']

final_submission = pd.concat([test_data['PassengerId'], y_pred], axis=1)
final_submission.to_csv(os.getcwd() + '\\Titanic\\FinalSubmission.csv', index=False)
