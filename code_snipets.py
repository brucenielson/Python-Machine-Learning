# This is a file where I put useful code snippets I don't intend to use but don't want to forget how to do them


"""
def fix_missing_values(X, column_name):
    # Old way I created a separate minor feature that the column was missing than set it to -1
    X['Has '+ column_name] = X.apply(lambda x: ~np.isnan(x[column_name]), axis=1)
    X[column_name] = X[column_name].fillna(-1.0)
    # But now, I'm going to fill in the mean value if it's missing using the imputer, so this is no longer used
    return X
"""

