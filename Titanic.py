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
# from importlib import reload


# Start log file
# logfile=os.getcwd()+'\\titanic.log'
# log.basicConfig(filename=logfile,level=log.INFO)
# log.info("Starting Titanic Training on " + str(time.strftime("%c")))


def munge_data(data):
    # X = pd.concat([train_data.ix[:,0:1], train_data.ix[:,2:3], train_data.ix[:,4:8], train_data.ix[:,9:]], axis=1) # .ix allows you to slice using labels and position, and concat pieces them back together. Not best way to do this.
    X = data[['PassengerId', 'Pclass', 'Sex', 'Age',  'SibSp', 'Parch', 'Fare', 'Embarked']]
    if 'Survived' in data:
        y = data['Survived']
    else:
        y = None

    cols_to_transform = ['Sex', 'Embarked']
    X = pd.get_dummies(X, columns = cols_to_transform )
    X['Age'] = X['Age'].fillna(-1)
    X['Fare'] = X['Fare'].fillna(-1)
    return X, y



def train_titanic_tree(X, y):
    tree = DecisionTreeClassifier(criterion='entropy', max_depth=5)
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