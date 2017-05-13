# http://quantsoftware.gatech.edu/MC3-Project-1
import random
import pandas as pd
import Titanic
import os
import math
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
        return build_tree(data)

    #def query(self, Xtest):


def build_tree(data, tree_type="random"):
    # See http://quantsoftware.gatech.edu/images/4/4e/How-to-learn-a-decision-tree.pdf
    #build_tree(data)
    # if data.shape[0] == 1:    return [leaf, data.y, NA, NA]
    # If only one row left, it's a leaf by default
    if len(data) == 1:
        return ["leaf", data.iloc[0, -1], -1, -1]

    #if all    data.y    same:    return [leaf, data.y, NA, NA]
    # Check to see if final column, i.e. Y, is all the same value. If so, terminate.
    if len(data.ix[:, -1].unique()) == 1:
        return ["leaf", data.iloc[0, -1], -1, -1]

    else:
        #determine best or random feature i to split on
        if tree_type == 'random':
            split_feature = get_random_feature(data)
        elif tree_type == 'entropy':
            split_feature = get_best_feature(data)
        else:
            raise Exception("Invalid tree_type")

        # Find where to split the feature up into
        #SplitVal	=	(data[random,i]	+	data[random,i])	/	2
        split_val = get_split_val(data, split_feature)

        # Split data up and return
        # Split data up
        #leftree = build_tree(data[data[:, i] <= SplitVal])
        #righttree = build_tree(data[data[:, i] > SplitVal])
        left_data = data[data.loc[:, split_feature] <= split_val]
        right_data = data[data.loc[:, split_feature] > split_val]

        # If the split fails, create a leaf
        if len(data) == len(left_data) or len(data) == len(right_data):
            return ["leaf", data.iloc[-1].value_counts().max(), -1, -1]

        # The split is good, so call recursively
        left_tree = build_tree(left_data, tree_type=tree_type)
        right_tree = build_tree(right_data, tree_type=tree_type)

        # Create root
        # root = [i,	SplitVal,	1,	leftree.shape[0]	+	1]
        root = [split_feature, split_val, 1, len(left_tree) + 1]

        # Return tree
        return create_rows(root, left_tree, right_tree)



def get_split_val(data, split_feature):
    # SplitVal	=	(data[random,i]	+	data[random,i])	/	2
    rand_nbr1 = random.randint(0,len(data)-1)
    item1 = data.iloc[rand_nbr1][split_feature]
    rand_nbr2 = rand_nbr1
    while rand_nbr2 == rand_nbr1:
        rand_nbr2 = random.randint(0, len(data)-1)
    item2 = data.iloc[rand_nbr2][split_feature]
    return (item1 + item2) / 2


def create_rows(root, left, right):
    root = remove_nesting(root)
    left = remove_nesting(left)
    right = remove_nesting(right)
    ret = [root]
    if type(left[0]) == list:
        for i in range(0,len(left)):
            ret.append(left[i])
    else:
        ret.append(left)
    if type(right[0]) == list:
        for i in range(0,len(right)):
            ret.append(right[i])
    else:
        ret.append(right)
    return ret


def remove_nesting(tree):
    if len(tree) > 1:
        return tree
    else:
        return remove_nesting(tree[0])



def get_best_feature(data):

    return


def calc_class_entropy(data):
    # Get possible classes in this data set. Y is always last column.
    classes = data.iloc[:][-1].unique()
    # From Machine Learning by Mitchel p. 57: Entropy = sum for each possible classification; -proportion log2 proportion.
    # Collective Intelligence, p. 148
    total_rows = len(data)
    entropy = 0
    for item in classes:
        class_rows = data[data.loc[:][-1] == item].count()
        portion = class_rows/total_rows
        entropy += -portion * math.log(portion, 2)

    return entropy




def get_random_feature(data):
    feature_nbr = random.randint(0, len(data.columns.values[0:-1]) - 1)  # [0,-1] to drop Y value and then -1 because zero indexed
    split_feature = data.columns.values[feature_nbr]
    return split_feature



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
