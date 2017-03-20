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

