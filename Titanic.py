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
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression

#TODO: from PlotDecision import plot_decision_regions


# Start log file
# logfile=os.getcwd()+'\\titanic.log'
# log.basicConfig(filename=logfile,level=log.INFO)
# log.info("Starting Titanic Training on " + str(time.strftime("%c")))


def munge_data(train_data, test_data=None, reduced_columns = False, verbose=True, use_top=None):
    X_train = train_data[['Pclass', 'Sex', 'Age',  'SibSp', 'Parch', 'Fare', 'Embarked', 'Name', 'Cabin', 'Ticket']]
    if 'Survived' in train_data:
        y_train = train_data['Survived']
    else:
        y_train = None

    # Were we also passed the Test set for one hot encoding?
    if test_data is not None:
        X_test = test_data[['Pclass', 'Sex', 'Age',  'SibSp', 'Parch', 'Fare', 'Embarked', 'Name', 'Cabin', 'Ticket']]
        # Combine both sets for cleansing
        X_all = pd.concat([X_train, X_test], ignore_index=True)
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



    # Save with Cabin and Ticket
    X_temp = X_all

    # One Hot Encoding way
    if reduced_columns == True:
        cols_to_transform = ['Sex', 'Embarked', 'Title', 'Deck', 'TicketPre']
    else:
        cols_to_transform = ['Sex', 'Embarked', 'Title', 'Deck', 'TicketPre', 'Cabin', 'Ticket']
    # First create columns by one hot encoding data and additional data (which will contain train and test data)
    X_all = pd.get_dummies(X_all, columns = cols_to_transform )

    # Add Ticket back in
    X_all['Ticket'] = X_temp['Ticket']

    """
    # Try PCAing the mass of Ticket and Cabin fields down to a handful
    filter_col = [col for col in list(X_all) if col.startswith('Ticket_') or col.startswith('Cabin_')]
    X_all_cabin_ticket = X_all.loc[:,filter_col]
    n_components=100
    pca = PCA(n_components=n_components)
    pca_names = ['PCA'+str(num) for num in range(0,n_components)]
    X_pca = pd.DataFrame(pca.fit_transform(X_all_cabin_ticket), columns=pca_names)
    print("PCA Explained Variance Ratio")
    print(pca.explained_variance_ratio_)
    X_all = X_all.drop(filter_col, axis=1)
    X_all = X_all.join(X_pca)
    """

    # Fix using Imputer -- fill in with mean for columns with continuous values
    imp = preprocessing.Imputer(missing_values='NaN', strategy='mean', axis=0)
    X_all[['Fare', 'Adj Age']] = imp.fit_transform(X_all[['Fare', 'Adj Age']])

    # Temp add new features for regression
    #X_all['Age2'] = X_all['Adj Age']**2
    #X_all['Fare2'] = X_all['Fare']**2
    #X_all['FamilySize'] = X_all['SibSp'] + X_all['Parch']

    # Scale and center
    col_names = ['Adj Age', 'SibSp', 'Parch', 'Fare', 'Pclass']#, 'FamilySize', 'Age2', 'Fare2' ]
    features = X_all[col_names]
    scaler = StandardScaler().fit(features.values)
    features = scaler.transform(features.values)
    X_all[col_names] = features

    # Remove columns that contain duplicate info
    #title_cols = [col for col in list(X_all) if col.startswith('Title_')]
    #dup_cols = ['Sex_1']# + title_cols
    #X_all = X_all.drop(dup_cols, axis=1)

    # Trying Family Guess again
    # Select out only the training portion
    min_cols = list(X_all.columns.values)
    drop_cols = [col for col in list(min_cols) if col.startswith('Ticket_') or col.startswith('Cabin_')]
    min_cols = [col for col in min_cols if col not in drop_cols]

    X_temp_train = X_all[0:len(X_train)][min_cols]
    # Make initial guess
    clf = LogisticRegression()
    clf.fit(X_temp_train, y_train)

    # Predict based on train and test together
    X_temp = X_all[min_cols]
    y_pred_initial = clf.predict(X_temp)
    y_pred_initial_prob = clf.predict_proba(X_temp)

    # Save off predictions
    X_all['Survival Guess'] = y_pred_initial_prob[:,1]
    # Create columns for filling in family survival chance based on prediction for oldest family member
    X_all['Family Survival Guess'] = -1
    # Now group familes and predict chances of parent surviving
    X_all['Family Survival Guess'] = X_all.apply(lambda x: calculate_parent_survival_factor(X_all, x), axis=1)
    #print(X_all.apply(lambda x: calculate_parent_survival_factor(X_all, x, le2), axis=1))
    X_all = X_all.drop('Survival Guess', axis=1)
    #X_all = X_all.drop('Family Survival Guess', axis=1)
    X_all = X_all.drop('Ticket', axis=1)

    # returns statistics
    y_pred_initial = y_pred_initial[0:len(X_train)]
    print("")
    print("Results of Predict - Initial Guess:")
    print('Misclassified train samples: %d' % (y_train != y_pred_initial).sum())
    print('Accuracy of train set: %.2f' % accuracy_score(y_train, y_pred_initial))



    # Now split train and test apart again
    X_train = X_all[0:len(X_train)]
    X_test = X_all[len(X_train):len(X_all)]
    #X_train = X_train[['Title_Mr', 'Fare', 'Pclass_3', 'Sex_0', 'Deck_C']]
    #X_test = X_test[['Title_Mr', 'Fare', 'Pclass_3', 'Sex_0', 'Deck_C']]

    # Calculate Correlations
    X_corr = X_train.copy(deep=True)
    X_corr['Survived'] = y_train.copy(deep=True)
    correlations = X_corr.corr()['Survived'].to_frame()
    correlations['Correlation'] = abs(correlations['Survived'])
    correlations = correlations.sort_values('Correlation', ascending=False)

    if use_top != None:
        # Only use top features
        if type(use_top) == int:
            correlations = correlations[0:use_top]['Survived']
        elif type(use_top) == float:
            correlations = correlations[correlations['Correlation'] >= use_top]['Survived']
        top_cols = list(correlations.index)
        top_cols.remove('Survived')
        X_train = X_train[top_cols]
        X_test = X_test[top_cols]

    # Do we want to show correlations?
    if verbose:
        print("***")
        print("Top Correlations:")
        print(correlations)


    if verbose:
        print("# of Columns:")
        print(len(X_train.columns))

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



def calculate_parent_survival_factor(X, person):
    parent = get_parent(X, person)
    if not parent.empty:
        val = float(parent['Survival Guess'])
    else:
        val = 0
    return val



def get_parent(X, person):
    ticket = person['Ticket']
    family = X.loc[(X['Ticket'] == ticket)]
    highest_age = -1.0
    highest_age_person = family.iloc[0]
    for i, row  in family.iterrows():
        if row['Adj Age'] > highest_age:
            highest_age_person = row
            highest_age = row['Adj Age']

    return highest_age_person



def titanic():

    # Read in training data
    trainfile = os.getcwd() + '\\Titanic\\train.csv'
    train_data = pd.read_csv(trainfile)
    # Grab Test data now so that we can build a proper one hot encoded data set
    # Test data often has options not in training data, so we have to review both together
    # If we want to one hot encoded things correctly
    testfile = os.getcwd() + '\\Titanic\\test.csv'
    test_data = pd.read_csv(testfile)

    # Now munge the train data, but include test data so we get consistent one hot encoding
    X_train, y_train, X_test = munge_data(train_data, test_data=test_data, reduced_columns=False, use_top=0.01)
    # Save out training data for bug fixing
    X_train.to_csv(os.getcwd() + '\\Titanic\\CheckData.csv', index=False)
    # Save out transformed Test Data for bug fixing
    X_test.to_csv(os.getcwd() + '\\Titanic\\Xtest.csv')



    """
    best_features = ml_helper.get_best_recursive_features(X_train, y_train, logistic_regression=True, random_forest = True, decision_tree = True, cv=10, create_graph=True)
    print(best_features)
    # Use only best features
    X_train = X_train[best_features['Logistic Regression']]
    X_test = X_test[best_features['Logistic Regression']]
    return
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

    # Create PairPlot graph
    #import seaborn
    #seaborn.set(style='whitegrid', context='notebook')
    #top_data = pd.concat([y_train, X_train.ix[:,0:7]], axis=1)
    #seaborn.pairplot(top_data)
