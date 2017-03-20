# This is a file where I put useful code snippets I don't intend to use but don't want to forget how to do them


"""
def fix_missing_values(X, column_name):
    # Old way I created a separate minor feature that the column was missing than set it to -1
    X['Has '+ column_name] = X.apply(lambda x: ~np.isnan(x[column_name]), axis=1)
    X[column_name] = X[column_name].fillna(-1.0)
    # But now, I'm going to fill in the mean value if it's missing using the imputer, so this is no longer used
    return X
"""

"""
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

"""
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
