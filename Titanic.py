import EnsembleClassifier as classifier
import pandas as pd
import numpy as np
import os
import logging as log
import re
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from importlib import reload
from sklearn.model_selection import cross_val_score # Note: What is cross_val_predict?
import MachineLearningHelper as ml_helper

#TODO: from PlotDecision import plot_decision_regions


# Start log file
# logfile=os.getcwd()+'\\titanic.log'
# log.basicConfig(filename=logfile,level=log.INFO)
# log.info("Starting Titanic Training on " + str(time.strftime("%c")))


def munge_data(train_data, test_data=None, show_corr = False, reduced_columns = False):
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
    X_all = X_all.drop('Age', axis=1)

    # Label Encoder way
    le = preprocessing.LabelEncoder()
    # Label Encode cabin
    le.fit(X_all.Cabin.unique())
    X_all['Cabin'] = le.transform(X_all['Cabin'])
    # Label Encode Sex
    le.fit(X_all.Sex.unique())
    X_all['Sex'] = le.transform(X_all['Sex'])
    # Label Encode Title
    #le.fit(X_all.Title.unique())
    #X_all['Title'] = le.transform(X_all['Title'])
    # Label Encode Deck
    #le.fit(X_all.Deck.unique())
    #X_all['Deck'] = le.transform(X_all['Deck'])
    # Label Encode Embarked
    #le.fit(X_all.Embarked.unique())
    #X_all['Embarked'] = le.transform(X_all['Embarked'])
    # Label Encode Ticket Prefix
    #le.fit(X_all.TicketPre.unique())
    #X_all['TicketPre'] = le.transform(X_all['TicketPre'])
    # Label Encode ticket
    le.fit(X_all.Ticket.unique())
    X_all['Ticket'] = le.transform(X_all['Ticket'])

    # To get down to fewer columns if desired
    if reduced_columns == True:
        X_all = X_all.drop('Ticket', axis=1)
        X_all = X_all.drop('Cabin', axis=1)

    # One Hot Encoding way
    if reduced_columns == True:
        cols_to_transform = ['Pclass', 'Sex', 'Embarked', 'Title', 'Deck', 'TicketPre'] #, 'Cabin', 'Ticket']
    else:
        cols_to_transform = ['Pclass', 'Sex', 'Embarked', 'Title', 'Deck', 'TicketPre', 'Cabin', 'Ticket']
    # First create columns by one hot encoding data and additional data (which will contain train and test data)
    X_all = pd.get_dummies(X_all, columns = cols_to_transform )

    # Fix using Imputer -- fill in with mean for columns with continuous values
    imp = preprocessing.Imputer(missing_values='NaN', strategy='mean', axis=0)
    X_all[['Fare', 'Adj Age']] = imp.fit_transform(X_all[['Fare', 'Adj Age']])

    # Temp add new features for regression
    X_all['Age2'] = X_all['Adj Age']**2
    X_all['Fare2'] = X_all['Fare']**2

    # Scale and center
    col_names = ['Adj Age', 'SibSp', 'Parch', 'Age2', 'Fare', 'Fare2' ]
    features = X_all[col_names]
    scaler = StandardScaler().fit(features.values)
    features = scaler.transform(features.values)
    X_all[col_names] = features

    # Now split train and test apart again
    X_train = X_all[0:len(X_train)]
    X_test = X_all[len(X_train):len(X_all)]

    print("# of Columns:")
    print(len(X_train.columns))

    # Do we want to show correlations?
    if show_corr == True:
        X_corr = X_train.copy(deep=True)
        X_corr['Survived'] = y_train.copy(deep=True)
        print(X_corr.corr()['Survived'])

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






# Read in training data
trainfile = os.getcwd() + '\\Titanic\\train.csv'
train_data = pd.read_csv(trainfile)
# Grab Test data now so that we can build a proper one hot encoded data set
# Test data often has options not in training data, so we have to review both together
# If we want to one hot encoded things correctly
testfile = os.getcwd() + '\\Titanic\\test.csv'
test_data = pd.read_csv(testfile)

# Now munge the train data, but include test data so we get consistent one hot encoding
X_train, y_train, X_test = munge_data(train_data, test_data=test_data, show_corr=True, reduced_columns=True)

# Save out training data for bug fixing
X_train.to_csv(os.getcwd() + '\\Titanic\\CheckData.csv', index=False)
# Save out transformed Test Data for bug fixing
X_test.to_csv(os.getcwd() + '\\Titanic\\Xtest.csv')



"""
best_features = ml_helper.get_best_recursive_features(X_train, y_train, logistic_regression=True, random_forest = True, decision_tree = True, cv=100, create_graph=True)
# Use only best features
X_train = X_train[best_features['Logistic Regression']]
X_test = X_test[best_features['Logistic Regression']]
exit()
"""

# weights = [lr, svc, knn, rfc, nb]
clf = classifier.train_ensemble_classifier(X_train, y_train, weights = [1, 1, 1, 1, 0], grid_search=False,
                                cv=10, persist_name="TitanicParams", use_persisted_values=True)

y_pred = clf.predict(X_train)
# returns statistics
print("")
print("Results of Predict:")
print('Misclassified train samples: %d' % (y_train != y_pred).sum())
print('Accuracy of train set: %.2f' % accuracy_score(y_train, y_pred))


# Oops, cross validation has to run the whole thing multiple times!
# Try Kfold Cross Validation and get a more realistic score
scores = cross_val_score(estimator=clf, X=X_train, y=y_train, cv=10)
print("")
print("Results of Cross Validation:")
print('CV accuracy: %.3f +/- %.3f' % (np.mean(scores), np.std(scores)))


# Now predict using Test Data
y_pred = clf.predict(X_test)
y_pred = pd.DataFrame(y_pred)
y_pred.columns = ['Survived']

final_submission = pd.concat([test_data['PassengerId'], y_pred], axis=1)
final_submission.to_csv(os.getcwd() + '\\Titanic\\FinalSubmission.csv', index=False)