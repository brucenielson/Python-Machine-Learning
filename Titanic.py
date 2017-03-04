import pandas as pd
import numpy as np
import os
import logging as log
import time
import sklearn
from PlotDecision import plot_decision_regions
from sklearn import preprocessing
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
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
    X = data[['Pclass', 'Sex', 'Age',  'SibSp', 'Parch', 'Fare', 'Embarked', 'Name', 'Cabin']]
    if 'Survived' in data:
        y = data['Survived']
    else:
        y = None

    # Out of name and cabin, create title and deck
    # For name we just want "Master", "Mr", "Miss", "Mrs" etc
    # For cabin we want just the deck letter
    X['Title'] = X.apply(lambda x: name_to_title(x['Name']), axis=1)
    X['Deck'] = X.apply(lambda x: cabin_to_deck(x['Cabin']), axis=1)
    X['Adj Age'] = X.apply(lambda x: x['Age'] if ~np.isnan(x['Age']) else fix_age(x['Title']), axis=1)
    X['Embarked'] = X.apply(lambda x: x['Embarked'] if (pd.isnull(x['Embarked']) == False) else "NA", axis=1)
    # Now Drop Name and Cabin because we no longer need them
    X = X.drop('Name', axis=1)
    X = X.drop('Cabin', axis=1)
    X = X.drop('Age', axis=1)


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
    # Old way
    #cols_to_transform = ['Sex', 'Embarked', 'Title', 'Deck']
    #X = pd.get_dummies(X, columns = cols_to_transform )

    # Fix using Imputer -- fill in with mean
    imp = preprocessing.Imputer(missing_values='NaN', strategy='mean', axis=0)
    X_imputed = imp.fit_transform(X)
    X = pd.DataFrame(X_imputed, columns=X.columns)

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
    X = X.drop('Deck', axis=1) #165
    X = X.drop('Parch', axis=1) #165
    X = X.drop('Title', axis=1) #168

    return X, y


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



"""
def fix_missing_values(X, column_name):
    # Old way I created a separate minor feature that the column was missing than set it to -1
    X['Has '+ column_name] = X.apply(lambda x: ~np.isnan(x[column_name]), axis=1)
    X[column_name] = X[column_name].fillna(-1.0)
    # But now, I'm going to fill in the mean value if it's missing using the imputer, so this is no longer used
    return X
"""




def train_titanic_tree(X, y):
    tree = DecisionTreeClassifier(criterion='entropy', max_depth=len(X.columns))
    tree.fit(X_train, y_train)
    return tree
    #forest = RandomForestClassifier(criterion='entropy', n_estimators=100, random_state=1,n_jobs=2)
    #forest.fit(X, y)
    #return forest


# Read in training data
trainfile = os.getcwd() + '\\Titanic\\train.csv'
train_data = pd.read_csv(trainfile)
X_train, y_train = munge_data(train_data)

X_train.to_csv(os.getcwd() + '\\Titanic\\CheckData.csv', index=False)

# How to return average age by each Title
#master = X_train.loc[(~pd.isnull(X_train['Age']))] # & (X_train['Title'] == 'Col')
#mean_by_group = master.groupby('Title').mean()
#print(mean_by_group['Age'])

tree = train_titanic_tree(X_train, y_train)
y_pred = tree.predict(X_train)

# returns statistics
print('Misclassified train samples: %d' % (y_train != y_pred).sum())
print('Accuracy of train set: %.2f' % accuracy_score(y_train, y_pred))
export_graphviz(tree, out_file=os.getcwd() + '\\Titanic\\TrainTree.dot', feature_names=X_train.columns.values, class_names=["Died", "Survived"])

# Now try test data
testfile = os.getcwd() + '\\Titanic\\test.csv'
test_data = pd.read_csv(testfile)
X_test, y_test = munge_data(test_data)
#X_test.to_csv(os.getcwd() + '\\Titanic\\Xtest.csv')


y_pred = tree.predict(X_test)
y_pred = pd.DataFrame(y_pred)
y_pred.columns = ['Survived']

final_submission = pd.concat([test_data['PassengerId'], y_pred], axis=1)
final_submission.to_csv(os.getcwd() + '\\Titanic\\FinalSubmission.csv', index=False)
