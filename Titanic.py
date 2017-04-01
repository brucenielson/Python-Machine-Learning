import EnsembleClassifier as classifier
import pandas as pd
import numpy as np
import os
import logging as log
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
from sklearn.model_selection import cross_val_score # Note: What is cross_val_predict?
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# Start log file
# logfile=os.getcwd()+'\\titanic.log'
# log.basicConfig(filename=logfile,level=log.INFO)
# log.info("Starting Titanic Training on " + str(time.strftime("%c")))


def munge_data(train_data, test_data=None):
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
    # The seemingly less useful ones that really do make a difference but blow out # of columns
    #X_all = X_all.drop('Cabin', axis=1)
    #X_all = X_all.drop('Ticket', axis=1)


    # Label Encoder way
    le = preprocessing.LabelEncoder()
    # Label Encode cabin
    le.fit(X_all.Cabin.unique())
    X_all['Cabin'] = le.transform(X_all['Cabin'])
    # Label Encode Sex
    le.fit(X_all.Sex.unique())
    X_all['Sex'] = le.transform(X_all['Sex'])
    # Label Encode Title
    le.fit(X_all.Title.unique())
    X_all['Title'] = le.transform(X_all['Title'])
    # Label Encode Deck
    le.fit(X_all.Deck.unique())
    X_all['Deck'] = le.transform(X_all['Deck'])
    # Label Encode Embarked
    le.fit(X_all.Embarked.unique())
    X_all['Embarked'] = le.transform(X_all['Embarked'])
    # Label Encode Ticket Prefix
    le.fit(X_all.TicketPre.unique())
    X_all['TicketPre'] = le.transform(X_all['TicketPre'])
    # Label Encode ticket
    le.fit(X_all.Ticket.unique())
    X_all['Ticket'] = le.transform(X_all['Ticket'])


    # Fix using Imputer -- fill in with mean for columns with continuous values
    imp = preprocessing.Imputer(missing_values='NaN', strategy='mean', axis=0)
    X_all[['Fare', 'Adj Age']] = imp.fit_transform(X_all[['Fare', 'Adj Age']])
    #X_all = pd.DataFrame(X_imputed, columns=X_all.columns)

    # Save with Cabin and Ticket
    X_temp = X_all

    # One Hot Encoding way
    cols_to_transform = ['Pclass', 'Sex', 'Embarked', 'Title', 'Deck', 'TicketPre', 'Cabin', 'Ticket']
    # First create columns by one hot encoding data and additional data (which will contain train and test data)
    X_all = pd.get_dummies(X_all, columns = cols_to_transform )

    # Add Ticket back in
    X_all['Ticket'] = X_temp['Ticket']

    # Temp add new features for regression
    X_all['Age2'] = X_all['Adj Age']**2
    X_all['Fare2'] = X_all['Fare']**2

    # Scale and center
    col_names = ['Adj Age', 'SibSp', 'Parch', 'Age2', 'Fare', 'Fare2' ]
    features = X_all[col_names]
    scaler = StandardScaler().fit(features.values)
    features = scaler.transform(features.values)
    X_all[col_names] = features

    # Select out only the training portion
    X_temp_train = X_all[0:len(X_train)]
    #X_temp_train = X_temp_train.drop('Cabin', axis=1)
    X_temp_train = X_temp_train.drop('Ticket', axis=1)
    # Make initial guess
    clf = classifier.train_ensemble_classifier(X_temp_train, y_train, weights=[1, 1, 1, 1, 0], cv=10, persist_name="TitanicParams", use_persisted_values=True)

    # Predict based on train and test together
    X_temp = X_all
    #X_temp = X_temp.drop('Cabin', axis=1)
    X_temp = X_temp.drop('Ticket', axis=1)
    y_pred_initial = clf.predict(X_temp)

    # Save off predictions
    X_all['Survival Guess'] = y_pred_initial
    # Create columns for filling in family survival chance based on prediction for oldest family member
    X_all['Family Survival Guess'] = -1
    # Now group familes and predict chances of parent surviving
    X_all['Family Survival Guess'] = X_all.apply(lambda x: calculate_parent_survival_factor(X_all, x), axis=1)
    #print(X_all.apply(lambda x: calculate_parent_survival_factor(X_all, x, le2), axis=1))
    X_all = X_all.drop('Survival Guess', axis=1)
    #X_all = X_all.drop('Family Survival Guess', axis=1)
    #X_all = X_all.drop('Cabin', axis=1)
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

    print("")
    print("# of Columns:")
    print(len(X_train.columns))
    #print(X_train)

    #X_train = X_train[['Family Survival Guess']]
    #X_test = X_test[['Family Survival Guess']]
    #X_train = X_train[['SibSp', 'Parch', 'Fare', 'Adj Age', 'Pclass_1', 'Pclass_2', 'Pclass_3', 'Sex_0', 'Sex_1', 'Family Survival Guess']]
    #X_test = X_test[['SibSp', 'Parch', 'Fare', 'Adj Age', 'Pclass_1', 'Pclass_2', 'Pclass_3', 'Sex_0', 'Sex_1', 'Family Survival Guess']]

    return X_train, y_train, X_test



def calculate_parent_survival_factor(X, person):
    parent = get_parent(X, person)
    if not parent.empty:
        val = int(parent['Survival Guess'])
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
X_train, y_train, X_test = munge_data(train_data, test_data=test_data)

#X_train = X_train[['Cabin_E58', 'TicketPre_C', 'Ticket_7552', 'Ticket_345764', 'Ticket_364849', 'Cabin_E44', 'Ticket_A/5. 10482', 'Cabin_D33', 'Ticket_347071', 'Ticket_29105', 'Ticket_382649', 'Ticket_PC 17611', 'Ticket_315098', 'Ticket_C 17369', 'Cabin_A10', 'Ticket_113773', 'Ticket_2678', 'Cabin_C95', 'Ticket_13049', 'Ticket_11967', 'TicketPre_STONO', 'Ticket_345779', 'Ticket_PC 17473', 'Ticket_36967', 'Ticket_347470', 'Cabin_C104', 'Ticket_2663', 'Ticket_29106', 'Cabin_C52', 'Ticket_2699', 'Ticket_386525', 'Ticket_113804', 'Ticket_244278', 'Ticket_SOTON/O.Q. 392078', 'Ticket_PC 17483', 'Ticket_A/5. 3337', 'Ticket_3101278', 'Ticket_A/5. 3336', 'Cabin_C124', 'Ticket_113781', 'Ticket_113503', 'Ticket_7546', 'Ticket_STON/O 2. 3101285', 'Ticket_W./C. 6609', 'Cabin_E63', 'Ticket_315084', 'Ticket_11753', 'Deck_NA', 'Ticket_349909', 'Cabin_C106', 'Ticket_A/5 3540', 'Ticket_111426', 'Ticket_19943', 'Ticket_2627', 'Ticket_343095', 'Ticket_250651', 'Ticket_347080', 'Ticket_SW/PP 751', 'Ticket_347077', 'Ticket_13213', 'Ticket_244270', 'Ticket_W./C. 6608', 'Ticket_4138', 'Ticket_PC 17485', 'Ticket_110465', 'Ticket_PC 17558', 'Cabin_C47', 'TicketPre_WEP', 'Cabin_A19', 'Ticket_248698', 'Ticket_7553', 'Cabin_C111', 'Ticket_STON/O2. 3101271', 'Cabin_NA', 'Ticket_237798', 'Ticket_PC 17474', 'Deck_C', 'Cabin_C148', 'Title_Mr', 'Cabin_C49', 'Ticket_111428', 'Ticket_347083', 'Ticket_347085', 'Ticket_SOTON/O.Q. 392087', 'Deck_D', 'Cabin_D6', 'Ticket_237671', 'Ticket_113788', 'Ticket_248747', 'Cabin_A31', 'Ticket_367228', 'Cabin_D48', 'TicketPre_WC', 'Ticket_CA 2144', 'Ticket_347088', 'Ticket_330935', 'Cabin_D45', 'Cabin_E8', 'Ticket_2691', 'Ticket_19947', 'Ticket_113786', 'Ticket_13567', 'Cabin_B86', 'Ticket_113051', 'Ticket_19988', 'Ticket_W./C. 14258', 'Ticket_364846', 'Ticket_113760', 'Ticket_364850', 'Ticket_STON/O 2. 3101286', 'Ticket_382651', 'Cabin_D19', 'Cabin_B20', 'Ticket_S.O./P.P. 3', 'Ticket_11668', 'Ticket_113055', 'Ticket_SOTON/OQ 392089', 'TicketPre_SOPP', 'Deck_G', 'Cabin_C92', 'Ticket_16988', 'Ticket_349240', 'Title_Master', 'Ticket_11774', 'Ticket_2668', 'Ticket_363291', 'Ticket_SC/PARIS 2146', 'Ticket_4135', 'Ticket_STON/O2. 3101290', 'Ticket_STON/O 2. 3101289', 'Ticket_312991', 'Title_NA', 'Ticket_111320', 'Ticket_PC 17595', 'Ticket_SC/Paris 2163', 'Cabin_C22 C26', 'Cabin_A20', 'Pclass_1', 'Ticket_4134', 'Ticket_5727', 'TicketPre_PP', 'Sex_male', 'Ticket_PP 9549', 'Cabin_D26', 'Ticket_2908', 'Ticket_3101281', 'Ticket_365226', 'Ticket_2665', 'Ticket_343120', 'Ticket_S.O.C. 14879', 'Ticket_PC 17476', 'Ticket_113806', 'Cabin_B41', 'Ticket_3101295', 'Ticket_CA. 2314', 'Ticket_17453', 'Ticket_PC 17593', 'Cabin_A6', 'Ticket_347081', 'Ticket_345773', 'Ticket_11751', 'Ticket_17463', 'Ticket_350034', 'Ticket_2651', 'Embarked_S', 'Pclass_2', 'Title_Rev', 'Cabin_A26', 'Cabin_D30', 'Ticket_349237', 'Ticket_347742', 'Ticket_2620', 'Ticket_113501', 'Ticket_364516', 'Ticket_364848', 'Ticket_19952', 'Ticket_111427', 'Cabin_C126', 'Ticket_367230', 'Ticket_345763', 'Cabin_E46', 'Ticket_239865', 'Ticket_110564', 'TicketPre_LINE', 'Ticket_112379', 'Ticket_1601', 'Ticket_27042', 'Cabin_B58 B60', 'Ticket_4133', 'Title_Mrs', 'Ticket_350417', 'Ticket_19996', 'Ticket_347054', 'Ticket_345774', 'Ticket_330909', 'Ticket_2677', 'Cabin_E25', 'Ticket_C.A. 2315', 'Ticket_367226', 'TicketPre_FCC', 'Ticket_35281', 'Cabin_C82', 'Sex_female', 'Ticket_C.A. 2673', 'Cabin_C93', 'Ticket_4136', 'Ticket_2653', 'Ticket_347073', 'Ticket_LINE', 'Ticket_345572', 'Cabin_G6', 'Ticket_382652', 'Ticket_315096', 'Ticket_PC 17758', 'Ticket_349245', 'Ticket_2661', 'Ticket_239853', 'Cabin_E10', 'Cabin_C70', 'Cabin_D35', 'Ticket_C.A. 37671', 'Ticket_347082', 'Cabin_D', 'Ticket_347087', 'Cabin_E17', 'Ticket_244252', 'Cabin_A23', 'Cabin_A32', 'Ticket_347089', 'Ticket_PC 17572', 'Ticket_370129', 'SibSp', 'Ticket_112277', 'Ticket_4137', 'Ticket_S.W./PP 752', 'Cabin_E24', 'Ticket_695', 'Ticket_350407', 'Pclass_3', 'Ticket_244373', 'Cabin_D56', 'Ticket_350406', 'Ticket_17474', 'Ticket_349236', 'Ticket_113794', 'Ticket_PC 17475', 'Ticket_220845', 'Ticket_350046', 'Ticket_2666', 'Adj Age', 'Cabin_B49', 'Ticket_350043', 'Ticket_7598', 'Ticket_STON/O 2. 3101269', 'Ticket_65306', 'TicketPre_SWPP', 'Ticket_W.E.P. 5734', 'Ticket_113050', 'Cabin_B102', 'Cabin_B38', 'Deck_E', 'Cabin_B96 B98', 'Ticket_111369', 'Ticket_2689', 'Cabin_E12', 'Ticket_113767', 'Ticket_STON/O 2. 3101288', 'Ticket_3101265', 'Cabin_D46', 'Cabin_E77']]
#X_test = X_test[['Cabin_E58', 'TicketPre_C', 'Ticket_7552', 'Ticket_345764', 'Ticket_364849', 'Cabin_E44', 'Ticket_A/5. 10482', 'Cabin_D33', 'Ticket_347071', 'Ticket_29105', 'Ticket_382649', 'Ticket_PC 17611', 'Ticket_315098', 'Ticket_C 17369', 'Cabin_A10', 'Ticket_113773', 'Ticket_2678', 'Cabin_C95', 'Ticket_13049', 'Ticket_11967', 'TicketPre_STONO', 'Ticket_345779', 'Ticket_PC 17473', 'Ticket_36967', 'Ticket_347470', 'Cabin_C104', 'Ticket_2663', 'Ticket_29106', 'Cabin_C52', 'Ticket_2699', 'Ticket_386525', 'Ticket_113804', 'Ticket_244278', 'Ticket_SOTON/O.Q. 392078', 'Ticket_PC 17483', 'Ticket_A/5. 3337', 'Ticket_3101278', 'Ticket_A/5. 3336', 'Cabin_C124', 'Ticket_113781', 'Ticket_113503', 'Ticket_7546', 'Ticket_STON/O 2. 3101285', 'Ticket_W./C. 6609', 'Cabin_E63', 'Ticket_315084', 'Ticket_11753', 'Deck_NA', 'Ticket_349909', 'Cabin_C106', 'Ticket_A/5 3540', 'Ticket_111426', 'Ticket_19943', 'Ticket_2627', 'Ticket_343095', 'Ticket_250651', 'Ticket_347080', 'Ticket_SW/PP 751', 'Ticket_347077', 'Ticket_13213', 'Ticket_244270', 'Ticket_W./C. 6608', 'Ticket_4138', 'Ticket_PC 17485', 'Ticket_110465', 'Ticket_PC 17558', 'Cabin_C47', 'TicketPre_WEP', 'Cabin_A19', 'Ticket_248698', 'Ticket_7553', 'Cabin_C111', 'Ticket_STON/O2. 3101271', 'Cabin_NA', 'Ticket_237798', 'Ticket_PC 17474', 'Deck_C', 'Cabin_C148', 'Title_Mr', 'Cabin_C49', 'Ticket_111428', 'Ticket_347083', 'Ticket_347085', 'Ticket_SOTON/O.Q. 392087', 'Deck_D', 'Cabin_D6', 'Ticket_237671', 'Ticket_113788', 'Ticket_248747', 'Cabin_A31', 'Ticket_367228', 'Cabin_D48', 'TicketPre_WC', 'Ticket_CA 2144', 'Ticket_347088', 'Ticket_330935', 'Cabin_D45', 'Cabin_E8', 'Ticket_2691', 'Ticket_19947', 'Ticket_113786', 'Ticket_13567', 'Cabin_B86', 'Ticket_113051', 'Ticket_19988', 'Ticket_W./C. 14258', 'Ticket_364846', 'Ticket_113760', 'Ticket_364850', 'Ticket_STON/O 2. 3101286', 'Ticket_382651', 'Cabin_D19', 'Cabin_B20', 'Ticket_S.O./P.P. 3', 'Ticket_11668', 'Ticket_113055', 'Ticket_SOTON/OQ 392089', 'TicketPre_SOPP', 'Deck_G', 'Cabin_C92', 'Ticket_16988', 'Ticket_349240', 'Title_Master', 'Ticket_11774', 'Ticket_2668', 'Ticket_363291', 'Ticket_SC/PARIS 2146', 'Ticket_4135', 'Ticket_STON/O2. 3101290', 'Ticket_STON/O 2. 3101289', 'Ticket_312991', 'Title_NA', 'Ticket_111320', 'Ticket_PC 17595', 'Ticket_SC/Paris 2163', 'Cabin_C22 C26', 'Cabin_A20', 'Pclass_1', 'Ticket_4134', 'Ticket_5727', 'TicketPre_PP', 'Sex_male', 'Ticket_PP 9549', 'Cabin_D26', 'Ticket_2908', 'Ticket_3101281', 'Ticket_365226', 'Ticket_2665', 'Ticket_343120', 'Ticket_S.O.C. 14879', 'Ticket_PC 17476', 'Ticket_113806', 'Cabin_B41', 'Ticket_3101295', 'Ticket_CA. 2314', 'Ticket_17453', 'Ticket_PC 17593', 'Cabin_A6', 'Ticket_347081', 'Ticket_345773', 'Ticket_11751', 'Ticket_17463', 'Ticket_350034', 'Ticket_2651', 'Embarked_S', 'Pclass_2', 'Title_Rev', 'Cabin_A26', 'Cabin_D30', 'Ticket_349237', 'Ticket_347742', 'Ticket_2620', 'Ticket_113501', 'Ticket_364516', 'Ticket_364848', 'Ticket_19952', 'Ticket_111427', 'Cabin_C126', 'Ticket_367230', 'Ticket_345763', 'Cabin_E46', 'Ticket_239865', 'Ticket_110564', 'TicketPre_LINE', 'Ticket_112379', 'Ticket_1601', 'Ticket_27042', 'Cabin_B58 B60', 'Ticket_4133', 'Title_Mrs', 'Ticket_350417', 'Ticket_19996', 'Ticket_347054', 'Ticket_345774', 'Ticket_330909', 'Ticket_2677', 'Cabin_E25', 'Ticket_C.A. 2315', 'Ticket_367226', 'TicketPre_FCC', 'Ticket_35281', 'Cabin_C82', 'Sex_female', 'Ticket_C.A. 2673', 'Cabin_C93', 'Ticket_4136', 'Ticket_2653', 'Ticket_347073', 'Ticket_LINE', 'Ticket_345572', 'Cabin_G6', 'Ticket_382652', 'Ticket_315096', 'Ticket_PC 17758', 'Ticket_349245', 'Ticket_2661', 'Ticket_239853', 'Cabin_E10', 'Cabin_C70', 'Cabin_D35', 'Ticket_C.A. 37671', 'Ticket_347082', 'Cabin_D', 'Ticket_347087', 'Cabin_E17', 'Ticket_244252', 'Cabin_A23', 'Cabin_A32', 'Ticket_347089', 'Ticket_PC 17572', 'Ticket_370129', 'SibSp', 'Ticket_112277', 'Ticket_4137', 'Ticket_S.W./PP 752', 'Cabin_E24', 'Ticket_695', 'Ticket_350407', 'Pclass_3', 'Ticket_244373', 'Cabin_D56', 'Ticket_350406', 'Ticket_17474', 'Ticket_349236', 'Ticket_113794', 'Ticket_PC 17475', 'Ticket_220845', 'Ticket_350046', 'Ticket_2666', 'Adj Age', 'Cabin_B49', 'Ticket_350043', 'Ticket_7598', 'Ticket_STON/O 2. 3101269', 'Ticket_65306', 'TicketPre_SWPP', 'Ticket_W.E.P. 5734', 'Ticket_113050', 'Cabin_B102', 'Cabin_B38', 'Deck_E', 'Cabin_B96 B98', 'Ticket_111369', 'Ticket_2689', 'Cabin_E12', 'Ticket_113767', 'Ticket_STON/O 2. 3101288', 'Ticket_3101265', 'Cabin_D46', 'Cabin_E77']]


"""
best_features = classifier.get_best_features(X_train, y_train, logistic_regression=True, cv=100)
# Use only best features
X_train = X_train[best_features['Logistic Regression']]
X_test = X_test[best_features['Logistic Regression']]
"""

#X_train = X_train[['SibSp', 'Adj Age', 'Pclass_3', 'Sex_female', 'Pclass_3']]
#X_test = X_test[['SibSp', 'Adj Age', 'Pclass_3', 'Sex_female', 'Pclass_3']]

X_train1, X_test1, y_train1, y_test1 = train_test_split(X_train, y_train, test_size=0.33)


# weights = [lr, svc, knn, rfc, nb]
clf = classifier.train_ensemble_classifier(X_train1, y_train1, weights = [1, 1, 1, 1, 0], grid_search=False,
                                cv=10, persist_name="TitanicParams", use_persisted_values=False)

print("******************************************************")
y_pred = clf.predict(X_train1)
# returns statistics
print("")
print("Results of Predict:")
print('Misclassified train samples: %d' % (y_train1 != y_pred).sum())
print('Accuracy of train set: %.2f' % accuracy_score(y_train1, y_pred))


# Oops, cross validation has to run the whole thing multiple times!
# Try Kfold Cross Validation and get a more realistic score
scores = cross_val_score(estimator=clf, X=X_train1, y=y_train1, cv=10)
print("")
print("Results of Cross Validation:")
#print('CV accuracy scores: %s' % scores)
print('CV accuracy: %.3f +/- %.3f' % (np.mean(scores), np.std(scores)))



y_pred = clf.predict(X_test1)
# returns statistics
print("")
print("Results of Predict:")
print('Misclassified train samples: %d' % (y_test1 != y_pred).sum())
print('Accuracy of train set: %.2f' % accuracy_score(y_test1, y_pred))


# Oops, cross validation has to run the whole thing multiple times!
# Try Kfold Cross Validation and get a more realistic score
scores = cross_val_score(estimator=clf, X=X_test1, y=y_test1, cv=10)
print("")
print("Results of Cross Validation:")
#print('CV accuracy scores: %s' % scores)
print('CV accuracy: %.3f +/- %.3f' % (np.mean(scores), np.std(scores)))

print("******************************************************")
# Now use test set





# Save out training data for bug fixing
X_train.to_csv(os.getcwd() + '\\Titanic\\CheckData.csv', index=False)
# Save out transformed Test Data for bug fixing
X_test.to_csv(os.getcwd() + '\\Titanic\\Xtest.csv')




# Now predict using Test Data
y_pred = clf.predict(X_test)
y_pred = pd.DataFrame(y_pred)
y_pred.columns = ['Survived']

final_submission = pd.concat([test_data['PassengerId'], y_pred], axis=1)
final_submission.to_csv(os.getcwd() + '\\Titanic\\FinalSubmission.csv', index=False)