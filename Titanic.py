import pandas as pd
import numpy as np
import os
import logging as log
import time
import sklearn
from PlotDecision import plot_decision_regions
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz
from sklearn.metrics import accuracy_score
import math
# from importlib import reload


# Start log file
# logfile=os.getcwd()+'\\titanic.log'
# log.basicConfig(filename=logfile,level=log.INFO)
# log.info("Starting Titanic Training on " + str(time.strftime("%c")))


def munge_data(data):
    # X = pd.concat([train_data.ix[:,0:1], train_data.ix[:,2:3], train_data.ix[:,4:8], train_data.ix[:,9:]], axis=1) # .ix allows you to slice using labels and position, and concat pieces them back together. Not best way to do this.
    X = data[['PassengerId', 'Pclass', 'Sex', 'Age',  'SibSp', 'Parch', 'Fare', 'Embarked', 'Name', 'Cabin']]
    if 'Survived' in data:
        y = data['Survived']
    else:
        y = None

    X = fix_missing_values(X, 'Age')
    X = fix_missing_values(X, 'Fare')

    # Pare Name and Cabin down
    # For name we just want "Master", "Mr", "Miss", "Mrs"
    # For cabin we want just the deck letter
    X['Name'] = X.apply(lambda x: name_to_title(x['Name']), axis=1)
    X['Cabin'] = X.apply(lambda x: cabin_to_deck(x['Cabin']), axis=1)

    cols_to_transform = ['Sex', 'Embarked', 'Name', 'Cabin']
    X = pd.get_dummies(X, columns = cols_to_transform )

    return X, y


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
    if cabin:
        # Get first character as that is the deck
        return cabin[0]
    else:
        return "NA"


def fix_missing_values(X, column_name):
    X['Has '+ column_name] = X.apply(lambda x: ~np.isnan(x[column_name]), axis=1)
    X[column_name] = X[column_name].fillna(-100)
    return X




def train_titanic_tree(X, y):
    print(X)
    tree = DecisionTreeClassifier(criterion='entropy', max_depth=7)
    tree.fit(X_train, y_train)
    return tree


# Read in training data
trainfile = os.getcwd() + '\\Titanic\\train.csv'
train_data = pd.read_csv(trainfile)
X_train, y_train = munge_data(train_data)
#print(X_train)

tree = train_titanic_tree(X_train, y_train)
y_pred = tree.predict(X_train)

# returns statistics
print('Misclassified train samples: %d' % (y_train != y_pred).sum())
print('Accuracy of train set: %.2f' % accuracy_score(y_train, y_pred))
export_graphviz(tree, out_file=os.getcwd() + '\\Titanic\\TrainTree.dot', feature_names=X_train.columns.values)

"""
# Now try test data
testfile = os.getcwd() + '\\Titanic\\test.csv'
test_data = pd.read_csv(testfile)
X_test, y_test = munge_data(test_data)
#X_test.to_csv(os.getcwd() + '\\Titanic\\Xtest.csv')
y_pred = tree.predict(X_test)
y_pred = pd.DataFrame(y_pred)
y_pred.columns = ['Survived']

final_submission = pd.concat([X_test['PassengerId'], y_pred], axis=1)
final_submission.to_csv(os.getcwd() + '\\Titanic\\FinalSubmission.csv', index=False)
"""