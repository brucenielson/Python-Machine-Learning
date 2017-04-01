# This is a file where I put useful code snippets I don't intend to use but don't want to forget how to do them


""" How to create minor features for NA
def fix_missing_values(X, column_name):
    # Old way I created a separate minor feature that the column was missing than set it to -1
    X['Has '+ column_name] = X.apply(lambda x: ~np.isnan(x[column_name]), axis=1)
    X[column_name] = X[column_name].fillna(-1.0)
    # But now, I'm going to fill in the mean value if it's missing using the imputer, so this is no longer used
    return X
"""

""" How to do pipelines
# How do pipelines with an RFECV and grid search too
    if weights[1] != 0:
        # Kernel SVC
        if (grid_search == True):
            basic_svc = SVC(probability=True)
        else:
            basic_svc = SVC(probability=True, kernel='linear')

        # Are we preforming recursive feature elimination?
        if (recursive_felim == True):
            rfecv = RFECV(estimator=basic_svc, step=1, scoring='accuracy', cv=cv)
            params = [('rfecv', rfecv), ('svc', basic_svc)]
            svc = Pipeline(params)
        else:
            svc = basic_svc

        # Are we doing a grid search this time (for SVC)?
        if (grid_search == True):
            # Grid search kernel SVCs
            svc = SVC(probability=True)
            C_range = 10. ** np.arange(-2, 2)
            kernel_options = ['poly'] #['poly', 'rbf', 'sigmoid']
            if param_grid == None:
                param_grid = dict(svc__C=C_range, svc__kernel = kernel_options)
            else:
                param_grid.update(dict(svc__C=C_range, svc__kernel = kernel_options))
        estimators.append(('SVC', svc))

    print ("Parameter Grid for Grid Search: ")
    print(param_grid)
"""

""" How do to cross validation
# Using Cross Validation Example
# Create train / test split
X_train_sub, X_test, y_train_sub, y_test = train_test_split(X_train, y_train, test_size=0.1)
# train classifier
clf, best_features = train_classifier2(X_train_sub, y_train_sub)#train_titanic_tree(X_train, y_train)
X_train_best = X_train[best_features]
X_train_sub = X_train_sub[best_features]
X_test = X_test[best_features]
y_pred_train = clf.predict(X_train_sub)
y_pred_cv = clf.predict(X_test)
# change X_test to match X_train shape

# returns statistics
print('Misclassified train samples: %d' % (y_train_sub != y_pred_train).sum())
print('Accuracy of train set: %.2f' % accuracy_score(y_train_sub, y_pred_train))

# returns statistics
print('Misclassified cross validation samples: %d' % (y_test != y_pred_cv).sum())
print('Accuracy of CV set: %.2f' % accuracy_score(y_test, y_pred_cv))
#export_graphviz(tree, out_file=os.getcwd() + '\\Titanic\\TrainTree.dot', feature_names=X_train.columns.values, class_names=["Died", "Survived"])

# Retrain whole set
# train classifier
clf, best_features = train_classifier2(X_train, y_train)#train_titanic_tree(X_train, y_train)
X_train = X_train[best_features]
y_pred = clf.predict(X_train)
# returns statistics
print('Misclassified train samples: %d' % (y_train != y_pred).sum())
print('Accuracy of train set: %.2f' % accuracy_score(y_train, y_pred))
"""


# How to return average age by each Title
#master = X_train.loc[(~pd.isnull(X_train['Age']))] # & (X_train['Title'] == 'Col')
#mean_by_group = master.groupby('Title').mean()
#print(mean_by_group['Age'])

""" How to do linear regression
# Use Linear Regression to determine chance of survival instead
# Create linear regression object
regr = LinearRegression()
# Train the model using the training sets
regr.fit(X_train, y_train)
# The coefficients
print('Coefficients: \n', regr.coef_)
# The mean squared error
print("Mean squared error: %.2f"
      % np.mean((regr.predict(X_train) - y_train) ** 2))
# Explained variance score: 1 is perfect prediction
print('Variance score: %.2f' % regr.score(X_train, y_train))
"""

#How to do concat
# X = pd.concat([train_data.ix[:,0:1], train_data.ix[:,2:3], train_data.ix[:,4:8], train_data.ix[:,9:]], axis=1) # .ix allows you to slice using labels and position, and concat pieces them back together. Not best way to do this.

""" How to do linear regression
# Select out only the training portion
X_train = X_all[0:len(X_train)]
# Use Linear Regression to determine chance of survival for use within families (i.e. same ticket or same cabin)
regr = LinearRegression()
# Train the model using the training sets
regr.fit(X_train, y_train)
y_pred_regr = regr.predict(X_all)
X_all['Survival Chance'] = y_pred_regr
# Create columns for filling in father and mother survival chance
X_all['Father Survival Chance'] = 0.0
X_all['Mother Survival Chance'] = 0.0
# Now group familes and predict chances of parent surviving
# X_all['Parent Survival Factor'] = calculate_parent_survival_factor(X_all, le2)
X_all['Father Survival Chance'] = X_all.apply(lambda x: calculate_parent_survival_factor(X_all, x, le2, 0), axis=1)
X_all['Mother Survival Chance'] = X_all.apply(lambda x: calculate_parent_survival_factor(X_all, x, le2, 1), axis=1)
# print(X_all.apply(lambda x: calculate_parent_survival_factor(X_all, x, le2), axis=1))
X_all = X_all.drop('Survival Chance', axis=1)
X_all = X_all.drop('Cabin', axis=1)
X_all = X_all.drop('Ticket', axis=1)
"""



""" How to do munge data with family survival guess
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
    X_all = X_all.drop('Family Survival Guess', axis=1)
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

"""


""" How to do train test split

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
"""