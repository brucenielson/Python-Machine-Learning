# http://quantsoftware.gatech.edu/MC3-Project-1
import random
import pandas as pd
import Titanic
import os
from importlib import reload

"""
Example of how to use:

import RTLearner as rt
learner = rt.RTLearner(leaf_size = 1, verbose = False) # constructor
learner.addEvidence(Xtrain, Ytrain) # training step
Y = learner.query(Xtest) # query

"""

class RTLearner(object):
    def __init__(self, leaf_size = 1, verbose = False):
        self.leaf_size = leaf_size
        self.verbos = verbose

    def addEvidence(self, Xtrain, Ytrain):
        # Combine X and Y together
        data = pd.concat([Xtrain, Ytrain], axis=1)
        return buildTree(data)

    #def query(self, Xtest):


def buildTree(data, treeType="random"):
    # See http://quantsoftware.gatech.edu/images/4/4e/How-to-learn-a-decision-tree.pdf
    #build_tree(data)
    # if data.shape[0] == 1:    return [leaf, data.y, NA, NA]
    # If only one row left, it's a leaf by default
    if len(data) == 1:
        return ["leaf", data.iloc[0, -1], -1, -1]

    #if all    data.y    same:    return [leaf, data.y, NA, NA]
    # Check to see if final column, i.e. Survived, is all the same value. If so, terminate.
    if len(data.ix[:, -1].unique()) == 1:
        return ["leaf", data.iloc[0, -1], -1, -1]

    else:
        #determine best feature i to split on
        if treeType == "random":
            #TODO: This seems to select Survived as a feature at times, but I can't see how that is possible
            splitFeatureNbr = random.randint(0,len(data.columns.values[0:-1])-1) # [0,-1] to drop Y value and then -1 because zero indexed
            splitFeature = data.columns[splitFeatureNbr]
            if splitFeature == 'Survived':
                print('here')
        elif treeType == 'entropy':
            splitFeature = determineBestSplit(data)
        else:
            raise Exception("treeType must be random or entropy")

        #SplitVal	=	(data[random,i]	+	data[random,i])	/	2
        #TODO: use correct splitVal
        featureData = data.ix[:, splitFeature]
        sum = data.ix[:, splitFeature].sum()
        count = data.ix[:, splitFeature].count()
        #TODO: This sometimes gets a divide by zero error. WHy would count ever by zero?
        if count == 0:
            print('here')
        splitVal = float(sum)/float(count)

        #leftree = build_tree(data[data[:, i] <= SplitVal])
        leftData = data[data.loc[:,splitFeature] <= splitVal]
        if len(leftData) > 0:
            leftTree = buildTree(leftData, treeType=treeType)
        else:
            leftTree = None

        #righttree = build_tree(data[data[:, i] > SplitVal])
        rightData = data[data.loc[:,splitFeature] > splitVal]
        if len(rightData) > 0:
            rightTree = buildTree(rightData, treeType=treeType)
        else:
            rightTree = None

        if not (leftTree == None or rightTree == None):
            #root = [i,	SplitVal,	1,	leftree.shape[0]	+	1]
            root = [splitFeature, splitVal, 1, len(leftTree)+1]
            #return (append(root, leftree, righttree))
            ret = []
            ret.append(root)
            ret.append(leftTree)
            ret.append(rightTree)
            return ret

        elif leftTree != None:
            return leftTree
        else:
            return rightTree




def determineBestSplit(Xtrain, Ytrain):
    return


"""
learners = []
kwargs = {"k":10}
for i in range(0,bags):
    learners.append(learner(**kwargs))
"""

# Read in training data
trainfile = os.getcwd() + '\\Titanic\\train.csv'
train_data = pd.read_csv(trainfile)
# Grab Test data now so that we can build a proper one hot encoded data set
# Test data often has options not in training data, so we have to review both together
# If we want to one hot encoded things correctly
testfile = os.getcwd() + '\\Titanic\\test.csv'
test_data = pd.read_csv(testfile)

X_train, y_train, X_test = Titanic.munge_data(train_data, test_data=test_data, show_corr=True, reduced_columns=True, verbose=False)

X_train = X_train[['Title_Mr', 'Fare2', 'Adj Age', 'Fare', 'Age2', 'Pclass_3', 'Sex_0', 'Pclass_2', 'Deck_C', 'SibSp']]
X_test = X_test[['Title_Mr', 'Fare2', 'Adj Age', 'Fare', 'Age2', 'Pclass_3', 'Sex_0', 'Pclass_2', 'Deck_C', 'SibSp']]

learner = RTLearner(leaf_size = 1, verbose = False) # constructor
result = learner.addEvidence(X_train, y_train) # training step
print(result)
