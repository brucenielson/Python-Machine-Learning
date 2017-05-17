# http://quantsoftware.gatech.edu/MC3-Project-1
import random
import pandas as pd
import numpy as np
import Titanic
import os
import math
from importlib import reload

"""
#Example of how to use RTLearner:

import RTLearner as rt
learner = rt.RTLearner(leaf_size = 1, verbose = False) # constructor
learner.addEvidence(Xtrain, Ytrain) # training step
Y = learner.query(Xtest) # query

"""

class RTLearner(object):
    def __init__(self, leaf_size = 1, verbose = False, tree_type="random", output_type="regression", is_continuous=True):
        self.leaf_size = leaf_size
        self.verbose = verbose
        self.tree_type = tree_type
        self.output_type = output_type
        self.input_type=is_continuous

    def addEvidence(self, Xtrain, Ytrain):
        # Combine X and Y together
        data = pd.concat([Xtrain, Ytrain], axis=1)
        # TODO: make it use ndarrays instead
        # ??data = np.concatenate([Xtrain, Ytrain], axis=1)
        return build_tree(data, leaf_size=self.leaf_size, tree_type=self.tree_type, output_type=self.output_type)

    #def query(self, Xtest):


"""
# How to use BagLearner:

import BagLearner as bl
learner = bl.BagLearner(learner = rt.RTLearner, kwargs = {"leaf_size":1}, bags = 20, boost = False, verbose = False)
learner.addEvidence(Xtrain, Ytrain)
Y = learner.query(Xtest)
"""

class BagLearner(object):
    def __init__(self, learner, kwargs, bags = 20, boost=False, verbose = False):
        self.bags = bags
        self.boost = boost
        self.verbose = verbose

    def addEvidence(selfself, Xtrain, Ytrain):
        # Combine X and Y together
        data = pd.concat([Xtrain, Ytrain], axis=1)
        # TODO: make it use ndarrays instead
        # #??data = np.concatenate([Xtrain, Ytrain], axis=1)






def build_tree(data, leaf_size=1, tree_type="random", output_type="classification"):
    # See http://quantsoftware.gatech.edu/images/4/4e/How-to-learn-a-decision-tree.pdf
    #build_tree(data)
    # if data.shape[0] == 1:    return [leaf, data.y, NA, NA]
    # If only one row left, it's a leaf by default
    if len(data) <= leaf_size:
        # number of rows left is under min leaf size, so make a leaf
        if output_type == "classification":
            # For classification trees, use the largest class of the Y value in the rows making up this leaf
            counts = dict(data.iloc[:, -1].value_counts())
            maximum_value = max(counts, key=counts.get)
            return ["leaf", maximum_value, -1, -1]
        elif output_type=="regression":
            # For regression trees, use the average of the Y value in the rows making up this leaf
            return ["leaf", data.iloc[:,-1].mean(), -1, -1]
        else:
            raise Exception("'output_type' must be either classification or regression.")

    #if all    data.y    same:    return [leaf, data.y, NA, NA]
    # Check to see if final column, i.e. Y, is all the same value. If so, terminate.
    if len(data.ix[:, -1].unique()) == 1:
        # Since all Y results are identical in the remaining rows, just return the first Y value for the rows
        return ["leaf", data.iloc[0, -1], -1, -1]

    else:
        # Main loop
        #determine best or random feature i to split on
        if tree_type == 'random':
            split_feature = get_random_feature(data)
            # Find where to split the feature up into by randomly selecting a feature and the mean of two rows for that feature
            # SplitVal	=	(data[random,i]	+	data[random,i])	/	2
            split_val = get_split_val(data, split_feature)

        elif tree_type == 'entropy' or tree_type == 'variance':
            # Both entropy (for trees with only discrete values) and variance (for trees with some continuous values) bot need to get the best feature by that criteria
            # Precondition: data must have rows
            split_feature, split_val = get_best_feature(data, tree_type=tree_type)
            # Post conditiion: split_feature and split_val give the best split in terms of entropy or variance
        else:
            raise Exception("'tree_type' must be random, entropy, or variance")

        # Split data up by puttings all of split_feature above split value into left tree and the rest in right tree
        # Split data up
        #leftree = build_tree(data[data[:, i] <= SplitVal])
        #righttree = build_tree(data[data[:, i] > SplitVal])
        left_data = data[data.loc[:, split_feature] >= split_val]
        right_data = data[data.loc[:, split_feature] < split_val]

        # If the split fails - i.e. all of the data went into one branch or neither branch has rows -- create a leaf
        data_size = len(data)
        left_size = len(left_data)
        right_size = len(right_data)
        if data_size == 0:
            raise Exception("Data contains no rows.")
        if left_size == 0 and right_size == 0:
            raise Exception("Both left tree and right tree can't both be zero")
        if left_size == data_size or right_size == data_size:
            if output_type == "classification":
                counts = dict(data.iloc[:,-1].value_counts())
                maximum_value = max(counts, key=counts.get)
                return ["leaf", maximum_value, -1, -1]
            elif output_type=='regression':
                return ["leaf", data.iloc[:,-1].mean(), -1, -1]
            else:
                raise Exception("'output_type' must be classification or regression")

        # Show statistics
        print("")
        print("")
        print("Split on:" + split_feature + " at " + str(split_val))
        print("Left Tree:")
        survived = len(left_data[left_data.iloc[:,-1]==1])
        total = len(data)
        percent = survived/len(left_data)
        print("Survived: "+str(survived)+"; % Survived: " + str(percent))
        print("Rows: " + str(len(left_data)))
        print("")
        print("Right Tree:")
        survived = len(right_data[right_data.iloc[:, -1] == 1])
        percent = survived/len(right_data)
        print("Survived: " + str(survived) + "; % Survived: " + str(percent))
        print("Rows: " + str(len(right_data)))
        print("*************************************")


        # The split is good -- there are rows in both left and right tree -- so call recursively
        left_tree = build_tree(left_data, leaf_size=leaf_size, tree_type=tree_type, output_type=output_type)
        right_tree = build_tree(right_data, leaf_size=leaf_size, tree_type=tree_type, output_type=output_type)

        # Create root
        # root = [i,	SplitVal,	1,	leftree.shape[0]	+	1]
        if (left_tree[0] != 'leaf'):
            root = [split_feature, split_val, 1, len(left_tree) + 1]
        else:
            # If left tree is a single node, then just add it and put the right size at 2
            root = [split_feature, split_val, 1, 2]

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



def get_best_feature(data, tree_type="random"):
    best_score = None
    best_feature = None
    best_split_val = None
    # Run through each feature and see which one has the best (highest) score
    feature_list = list(data.columns)
    del feature_list[-1]
    for feature in feature_list:
        # If continuous, use variance, otherwise use entropy
        if tree_type == "variance":
            scoref = calc_variance
        elif tree_type == "entropy":
            scoref = calc_entropy
        else:
            raise Exception("'tree_type' must be variance or entropy")

        score, split_val = calc_info_gain(scoref, data, feature)
        if best_score == None or score >= best_score:
            best_score = score
            best_feature = feature
            best_split_val = split_val

    if best_score <= 0 or best_score == None:
        print("here")
    return best_feature, best_split_val


def calc_entropy(data):
    if len(data)==0: return 0
    # Get possible classes in this data set. Y is always last column.
    classes = data.iloc[:,-1].unique()
    # From Machine Learning by Mitchel p. 57: Entropy = sum for each possible classification; -proportion log2 proportion.
    # Collective Intelligence, p. 148
    total_rows = len(data)
    entropy = 0
    for item in classes:
        class_rows = data[data.iloc[:,-1] == item].iloc[:,-1].count()
        proportion = class_rows/total_rows
        entropy += -proportion * math.log(proportion, 2)

    return entropy

def calc_variance(data):
    if len(data)==0: return 0
    values = [row[-1] for index, row in data.iterrows()]
    mean = sum(values) / len(data)
    variance = sum( [(value - mean)**2 for value in values] ) / len(data)
    return variance



def calc_variance_old(data, feature):
    current_variance = data.iloc[:, -1].var()
    # For each possible value, calculate it's variance if you split on this value
    values = data.loc[:, feature].unique()
    total_rows = len(data)
    best_variance = 0
    best_criteria = None

    for value in values:
        set1 = data[data.loc[:,feature] >= value]
        set2 = data[data.loc[:,feature] < value]
        variance1 = set1.iloc[:, -1].var()
        if pd.isnull(variance1): variance1 = 0
        variance2 = set2.iloc[:, -1].var()
        if pd.isnull(variance2): variance2 = 0
        proportion_1 = len(set1)/total_rows
        proportion_2 = 1-proportion_1
        new_variance = current_variance - (proportion_1* variance1) - (proportion_2* variance2)
        if new_variance >= best_variance:
            best_variance = current_variance
            best_criteria = value

    if best_variance <= 0 or best_criteria == None:
        print("here")
    return best_variance, best_criteria



def calc_info_gain(scoref, data, feature):
    if len(data) == 0: raise Exception("calc_info_gain requires that data not be empty")
    if feature == None: raise Exception("calc_info_gain requires that feature be not None")

    current_score = scoref(data)
    # Get possible options to split on for this feature
    values = data.loc[:,feature].unique()
    # From Machine Learning by Mitchel p. 58
    total_rows = len(data)
    # Initialize save best value
    best_value = None
    best_score = 0
    # For each possible value, calculate it's score
    for value in values:
        if type(value) == int or np.issubdtype(type(value), np.integer):
            set1 = data[data.loc[:,feature] == value]
            set2 = data[data.loc[:,feature] != value]
        elif type(value) == float or np.issubdtype(type(value), np.float):
            set1 = data[data.loc[:, feature] >= value]
            set2 = data[data.loc[:, feature] < value]
        else:
            raise Exception("Values need to be int or float. Please format.")
        proportion_1 = len(set1)/total_rows
        proportion_2 = 1-proportion_1
        score = current_score - (proportion_1 * scoref(set1)) - (proportion_2 * scoref(set2))
        if score >= best_score:
            best_value = value
            best_score = score

    return best_score, best_value



def get_random_feature(data):
    feature_nbr = random.randint(0, len(data.columns.values[0:-1]) - 1)  # [0,-1] to drop Y value and then -1 because zero indexed
    split_feature = data.columns.values[feature_nbr]
    return split_feature



"""
# Instantiate several learners with the parameters listed in kwargs
learners = []
kwargs = {"k":10}
for i in range(0,bags):
    learners.append(learner(**kwargs))
"""
def train_titanic():
    # Read in training data
    trainfile = os.getcwd() + '\\Titanic\\train.csv'
    train_data = pd.read_csv(trainfile)
    # Grab Test data now so that we can build a proper one hot encoded data set
    # Test data often has options not in training data, so we have to review both together
    # If we want to one hot encoded things correctly
    testfile = os.getcwd() + '\\Titanic\\test.csv'
    test_data = pd.read_csv(testfile)

    X_train, y_train, X_test = Titanic.munge_data(train_data, test_data=test_data, show_corr=True, reduced_columns=True, verbose=False)

    X_train = X_train[['Title_Mr', 'Adj Age', 'Fare', 'Pclass_3', 'Sex_0', 'Pclass_2', 'Deck_C', 'SibSp']]
    X_test = X_test[['Title_Mr', 'Adj Age', 'Fare', 'Pclass_3', 'Sex_0', 'Pclass_2', 'Deck_C', 'SibSp']]

    learner = RTLearner(leaf_size = 20, verbose = False, output_type="classification", tree_type="variance") # constructor
    result = learner.addEvidence(X_train, y_train) # training step
    print(result)

